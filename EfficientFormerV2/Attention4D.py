# MODIFIED: Attention4D class to handle dynamic input shapes
class Attention4D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        # Store original resolution for bias calculation, but don't rely on it for forward pass shape
        self.original_resolution = resolution

        if stride is not None:
            # self.resolution = math.ceil(resolution / stride) # This is also problematic, stride logic needs to be in forward pass
            self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                             nn.BatchNorm2d(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            # self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        # self.N = self.resolution ** 2 # <-- PROBLEM: This is fixed at init. REMOVED.
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2d(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2d(self.num_heads * self.d), )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(act_layer(),
                                  nn.Conv2d(self.dh, dim, 1),
                                  nn.BatchNorm2d(dim), )

        # --- ATTENTION BIAS: This part is tricky. It's pre-calculated and assumes a fixed resolution.
        # A simple fix is to only apply it if the dynamic resolution matches the original one.
        # A full fix requires re-calculating the bias indices dynamically, which is complex.
        points = list(itertools.product(range(self.original_resolution), range(self.original_resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N), persistent=False) # Use persistent=False for buffer

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
        
        # NEW: Get dynamic shape and token number
        B, C, H, W = x.shape
        N = H * W

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        
        # NEW: Apply bias only if resolution matches, to prevent crashing or incorrect logic.
        if H == self.original_resolution and W == self.original_resolution:
            bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
            attn = attn + bias
        
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)
        
        # NEW: Use dynamic shape for reshaping output
        out = x.transpose(2, 3).reshape(B, self.dh, H, W) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out

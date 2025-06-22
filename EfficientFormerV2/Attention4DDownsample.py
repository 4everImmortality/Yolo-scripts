# MODIFIED: Attention4DDownsample class to handle dynamic input shapes
class Attention4DDownsample(torch.nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 out_dim=None,
                 act_layer=None,
                 ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        # Store original resolution for bias calculation
        self.original_resolution = resolution
        self.original_resolution2 = math.ceil(self.original_resolution / 2)

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.original_resolution, self.original_resolution2)

        # self.N = self.resolution ** 2 <-- PROBLEM: REMOVED
        # self.N2 = self.resolution2 ** 2 <-- PROBLEM: REMOVED

        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2d(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2d(self.num_heads * self.d), )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim), )

        # --- ATTENTION BIAS: This part is tricky. It's pre-calculated and assumes a fixed resolution.
        points = list(itertools.product(range(self.original_resolution), range(self.original_resolution)))
        points_ = list(itertools.product(
            range(self.original_resolution2), range(self.original_resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.original_resolution / self.original_resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.original_resolution / self.original_resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        
        # NEW: Get dynamic N for k and v
        N_kv = H * W
        
        q = self.q(x)
        # NEW: Get dynamic shape and N for q
        B_q, C_q, H_q, W_q = q.shape
        N_q = H_q * W_q

        q = q.flatten(2).reshape(B, self.num_heads, -1, N_q).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, N_kv).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, N_kv).permute(0, 1, 3, 2)
        
        attn = (q @ k) * self.scale
        
        # NEW: Apply bias only if resolution matches
        if H == self.original_resolution and W == self.original_resolution:
            bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
            attn = attn + bias
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        
        # NEW: Use dynamic shape for reshaping output
        out = x.reshape(B, self.dh, H_q, W_q) + v_local

        out = self.proj(out)
        return out

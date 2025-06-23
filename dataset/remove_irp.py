import os
import argparse

def rename_files_in_directory(directory_path, dry_run=False):
    """
    Scans a directory and renames files containing '.irp.' 
    by removing the '.irp' part.
    
    Example: '123.irp.jpg' -> '123.jpg'
    """
    print(f"\nProcessing directory: {directory_path}")
    
    # 检查路径是否存在且是否为目录
    if not os.path.isdir(directory_path):
        print(f"  Error: '{directory_path}' is not a valid directory. Skipping.")
        return 0

    files_renamed_count = 0
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件名是否包含 '.irp.' 这个特定的错误模式
        if ".irp." in filename:
            # 构建旧的完整文件路径
            old_filepath = os.path.join(directory_path, filename)
            
            # 创建新的文件名，只替换第一个 '.irp.' 为 '.'
            # 使用 count=1 更安全，防止文件名中有多个 '.irp.' 的意外情况
            new_filename = filename.replace(".irp.", ".", 1)
            
            # 构建新的完整文件路径
            new_filepath = os.path.join(directory_path, new_filename)
            
            if dry_run:
                print(f"  [DRY RUN] Would rename: '{filename}' -> '{new_filename}'")
            else:
                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"  Renamed: '{filename}' -> '{new_filename}'")
                    files_renamed_count += 1
                except OSError as e:
                    print(f"  Error renaming '{filename}': {e}")
    
    if files_renamed_count == 0 and not dry_run:
        print("  No files needed renaming in this directory.")
        
    return files_renamed_count

def main():
    parser = argparse.ArgumentParser(
        description="Batch rename files by removing a superfluous '.irp' from their names.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('directories', nargs='+', help='One or more directory paths to process.')
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help="Simulate the renaming process without actually changing any files. Highly recommended to run this first."
    )
    
    args = parser.parse_args()

    if args.dry_run:
        print("--- DRY RUN MODE ---")
        print("No files will be changed. Simulating the process...\n")
    
    total_renamed = 0
    for directory in args.directories:
        total_renamed += rename_files_in_directory(directory, args.dry_run)
        
    print("\n--- Summary ---")
    if args.dry_run:
        print("Dry run complete. Review the changes above. If they look correct, run the script again without the --dry-run flag.")
    else:
        print(f"Renaming complete. Total files renamed: {total_renamed}")

if __name__ == "__main__":
    main()

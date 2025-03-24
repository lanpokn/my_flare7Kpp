import os
import argparse
import torch
import lpips
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_image(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)

def calculate_metrics(input_folder, gt_folder):
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    count = 0

    # LPIPS model (use AlexNet as default backbone)
    loss_fn = lpips.LPIPS(net='alex').cuda()

    # 获取两个文件夹的文件列表，并按文件名排序
    input_files = sorted(os.listdir(input_folder))
    gt_files = sorted(os.listdir(gt_folder))

    # 检查文件数量是否一致
    if len(input_files) != len(gt_files):
        print(f"Warning: Different number of files in folders ({len(input_files)} vs {len(gt_files)}). "
              "Only the first {min(len(input_files), len(gt_files))} pairs will be processed.")

    # 按顺序配对文件
    for input_filename, gt_filename in zip(input_files, gt_files):
        input_path = os.path.join(input_folder, input_filename)
        gt_path = os.path.join(gt_folder, gt_filename)

        input_img = load_image(input_path).cuda()
        gt_img = load_image(gt_path).cuda()

        # Convert images to numpy for PSNR and SSIM
        input_np = input_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = gt_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Compute PSNR
        psnr_value = psnr(gt_np, input_np, data_range=1.0)

        # Compute SSIM with dynamic win_size
        min_size = min(input_np.shape[0], input_np.shape[1])
        win_size = min(7, min_size if min_size % 2 != 0 else min_size - 1)  # Ensure odd win_size <= min_size
        ssim_value = ssim(gt_np, input_np, data_range=1.0, win_size=win_size, channel_axis=-1)

        # Compute LPIPS
        lpips_value = loss_fn(input_img, gt_img).item()

        psnr_total += psnr_value
        ssim_total += ssim_value
        lpips_total += lpips_value
        count += 1

    if count == 0:
        print("No files found in one or both folders.")
        return

    print(f'Average PSNR: {psnr_total / count:.4f}')
    print(f'Average SSIM: {ssim_total / count:.4f}')
    print(f'Average LPIPS: {lpips_total / count:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics.')
    parser.add_argument('--input', required=True, help='Path to input images')
    parser.add_argument('--gt', required=True, help='Path to ground truth images')
    args = parser.parse_args()

    calculate_metrics(args.input, args.gt)

if __name__ == '__main__':
    main()

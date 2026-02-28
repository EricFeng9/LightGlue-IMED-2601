import os
import cv2
import argparse
import glob
from gen_data_enhance import random_domain_augment_image

def get_next_index(directory):
    """获取目录下未被使用的最小可用编号(000-999)"""
    if not os.path.exists(directory):
        return 0
    existing_files = glob.glob(os.path.join(directory, "*.png"))
    existing_indices = []
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            # 提取前缀的数字
            idx = int(basename.split('_')[0])
            existing_indices.append(idx)
        except ValueError:
            continue
    
    for i in range(1000):
        if i not in existing_indices:
            return i
    return 1000

def process_image(image_path, save_dir, amount, prefix_name):
    """读取图像，进行指定次数的域随机化并保存"""
    if not os.path.exists(image_path):
        print(f"Error: 找不到图像 {image_path}")
        return

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取图像并转换为灰度图（LoFTR使用灰度图进行配准）
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 无法读取图像 {image_path}")
        return
    
    # 转换为灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"已将图像转换为灰度图")
    
    # 转回3通道以便进行域增强（域增强需要3通道图像）
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print(f"开始处理 {image_path}，计划生成 {amount} 张...")
    
    for _ in range(amount):
        next_idx = get_next_index(save_dir)
        if next_idx >= 1000:
            print("Error: 编号已达到 999 限制，停止生成。")
            break
            
        # 域增强
        aug_img = random_domain_augment_image(img)
        
        # 保存
        filename = f"{next_idx:03d}_{prefix_name}_aug.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, aug_img)
        print(f"已生成: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="域随机化测试脚本")
    parser.add_argument("--amount", type=int, default=20, help="每张图像进行随机化的次数 (默认: 20)")
    args = parser.parse_args()
    
    amount = args.amount
    
    # 使用脚本所在目录的相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "test_data_enhance")
    
    cf_path = os.path.join(base_dir, "cf_gen.png")
    fa_path = os.path.join(base_dir, "fa_gen.png")
    
    cf_save_dir = os.path.join(base_dir, "cf_results")
    fa_save_dir = os.path.join(base_dir, "fa_results")
    
    process_image(cf_path, cf_save_dir, amount, "cf")
    process_image(fa_path, fa_save_dir, amount, "fa")
    
    print("完成所有任务！")

if __name__ == "__main__":
    main()

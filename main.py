import json
import os
from PIL import Image

from output_helper.common import find_files_by_pattern_recursive, IMAGE_EXTENSIONS, get_random_color
from output_helper.excel import ExcelWorkbook
from signature_analyzer.lbp_patterns import calc_lbp_patterns
from signature_analyzer.utils import compare_hist


dataset_folder = r"C:\Users\ivpr0122\Documents\Personal\sign_data\test"

def calc_patterns():
    for dir_num in range(49, 70):
        signature_variations_dir = f"{dataset_folder}\\0{dir_num}"
        for image_filename in os.listdir(signature_variations_dir):
            _, file_ext = os.path.splitext(image_filename)
            if file_ext in IMAGE_EXTENSIONS:
                full_image_path = f"{signature_variations_dir}\\{image_filename}"
                if (True or not os.path.exists(full_image_path.replace(file_ext, ".json"))): # Remove 'True or' after debug
                    lpb_patterns = calc_lbp_patterns(Image.open(full_image_path))
                    with open(full_image_path.replace(file_ext, ".json"), 'w') as f:
                        json.dump(lpb_patterns, f)


def compare_and_write_to_excel():
    workbook = ExcelWorkbook('compare_hist.xlsx')

    current_excel_symbol = 1  # 'B' symbol
    current_excel_num = 2  # Second row

    for i_dir_num in range(49, 70):
        i_signature_variations_dir = f"{dataset_folder}\\0{i_dir_num}"
        i_all_files = find_files_by_pattern_recursive(i_signature_variations_dir, r'.*\.json')

        for i_file in i_all_files:
            current_excel_num = 2
            workbook.write(current_excel_symbol, 1, os.path.basename(i_file), color=get_random_color(i_dir_num))
            with open(i_file) as f:
                i_file_lbp_patterns = json.load(f)

            for j_dir_num in range(49, 70):
                j_signature_variations_dir = f"{dataset_folder}\\0{j_dir_num}"
                j_all_files = find_files_by_pattern_recursive(j_signature_variations_dir, r'.*\.json')

                for i, j_file in enumerate(j_all_files, current_excel_num):
                    with open(j_file) as f:
                        j_file_lbp_patterns = json.load(f)
                    distance = compare_hist(i_file_lbp_patterns, j_file_lbp_patterns)

                    workbook.write(0, i, os.path.basename(j_file), color=get_random_color(j_dir_num))
                    workbook.write(current_excel_symbol, i, distance)
                    current_excel_num += 1

            current_excel_symbol += 1

    workbook.dump()


if __name__ == '__main__':
    calc_patterns()
    # compare_and_write_to_excel()

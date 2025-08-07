import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random
import json
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import sys


# ドキュメントのパスを取得
def return_doc_path_list(doc_path_list = []):
    dir_path = '../kuzushiji-recognition/char_sep_datas'
    dir_list = [
        f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
    ]
    doc_path_list = []
    for doc_id in dir_list:
        doc_path_list.append(dir_path+'/'+doc_id)
    return doc_path_list

# ドキュメントIDに対して、ファイルパスを取得。
def return_file_path_list(doc_path):
    files_list = [
        f for f in os.listdir(doc_path+'/images') if os.path.isfile(os.path.join(doc_path+'/images',f))
    ]
    file_path_list = []
    for image_name in files_list:
        file_path_list.append(doc_path+'/images/'+image_name)
    return file_path_list

# 一つのファイルパス対する処理群
def return_image(file_path):
    image = np.array(Image.open(file_path))
    return image
def return_gray_image(file_path):
    gray_image = np.array(Image.open(file_path).convert('L'))
    return gray_image
def judge_brightness(file_path, threshold=110):
    gray_image = return_gray_image(file_path)
    if gray_image.mean() < threshold:
        return True #暗すぎたら飛ばす。
    else:
        return False
def judge_brightness_and_return_image(file_path):
    if judge_brightness(file_path):
        return None
    else:
        image = return_image(file_path)
        return image
    
# メインで実行する関数のフォーマット確認
def main_exe_for_one_image(file_path, procedure_for_one_image=None):
    image = judge_brightness_and_return_image(file_path)
    if image is None:
        print(f"Skipping {file_path.split('/')[-1]} due to brightness.")
        return None
    else:
        procedure_for_one_image(image,file_path)
def main_exe(procedure_for_one_image = None, testdata_doc_id = []):
    doc_path_list = return_doc_path_list(doc_path_list=testdata_doc_id)
    for doc_path in doc_path_list:
        file_path_list = return_file_path_list(doc_path)
        for file_path in file_path_list:
            if procedure_for_one_image != None:
                # 一枚の画像に対して、疑似的を文字を合成し、自己教師あり学習の正解データを作成する。
                main_exe_for_one_image(file_path=file_path, procedure_for_one_image=procedure_for_one_image)

# 指定した画像を白紙に戻す
def remove_ink(image: np.ndarray, show_flag=False, inpaint_radius=20) -> np.ndarray:
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 平滑化して濃淡を均一化
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 大津の二値化でマスクを生成
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # マスクを膨張させて文字領域を広げる
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaintingを使用して文字領域を周囲の色で埋める
    inpainted = cv2.inpaint(image, dilated_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return inpainted
def plt_show_image(image, title='Image'):
    try:
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except:
        print(f'{title} : {type(image)}')

# 文書の上下に余白をどれだけ作るか決定
def decide_upper_lower_coodinate(image: np.ndarray) -> tuple:
    char_region_rate1 = np.random.normal(loc=75,scale=7,size=(1))[0]
    while char_region_rate1 >= 100 or char_region_rate1 <= 0:
        char_region_rate1 = np.random.normal(loc=75,scale=7,size=(1))[0]
    char_region_rate2 = np.random.normal(loc=75,scale=7,size=(1))[0]
    while char_region_rate2 >= 100 or char_region_rate2 <= 0:
        char_region_rate2 = np.random.normal(loc=75,scale=7,size=(1))[0]
    length_region = int(image.shape[0] * char_region_rate1 / 100)
    width_region = int(image.shape[1] * char_region_rate2 / 100)
    upper_cood = int((image.shape[0]-length_region)/2)
    lower_cood = int(upper_cood + length_region)
    left_cood = int((image.shape[1]-width_region)/2)
    right_cood = int(left_cood + width_region)
    return upper_cood, lower_cood, left_cood, right_cood

# 何行含むか決定
def generate_uniform_integers(low=6, high=15, size=1):
    return np.random.randint(low, high + 1, size)[0]

# 正解データのためのキャンバスをnumpy.ndarrayで作成
def return_ground_truth_canvas(image):
    main_region = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
    main_affinity = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
    furi_region = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
    furi_affinity = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
    return main_region, main_affinity, furi_region, furi_affinity

# 注目する行の座標を計算する。
def focus_on_one_line(upper_cood, lower_cood, left_cood, right_cood, number_of_lines):
    line_width = int((right_cood - left_cood) / number_of_lines)
    focus_region_list = []
    for line_num in range(number_of_lines):
        focuse_left_cood = left_cood + line_num * line_width
        focuse_right_cood = left_cood + (line_num + 1) * line_width
        focus_region_list.append((upper_cood, lower_cood, focuse_left_cood, focuse_right_cood))
    return focus_region_list

# ランダムに画像を選択。
def convert_to_characters_path(file_path):
    return file_path.rsplit('/', 2)[0] + '/characters'
def select_random_folder(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if not folders:
        return None  
    return random.choice(folders)
def select_random_file(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return None  
    return random.choice(files)

def return_random_char(file_path, furi_mode = False):
    while True:
        # 文字のフォルダを取得
        characters_path = convert_to_characters_path(file_path)
        # ランダムにフォルダを選択
        selected_char = select_random_folder(characters_path)
        if selected_char is None:
            continue
        if furi_mode and not is_hiragana(selected_char):
            continue
        # ランダムにファイルを選択
        selected_image = select_random_file(characters_path + '/' + selected_char)
        if selected_image is None:
            continue 
        # 選択した画像のパスを取得
        selected_image_path = characters_path + '/' + selected_char + '/' + selected_image
        image = Image.open(selected_image_path)
        return image, selected_char
    
# 画像を適切なサイズにリサイズして、numpy配列に変換する関数
def resize_char_image_old(char_width, char_image):
    # Get the original dimensions of the image
    original_width, original_height = char_image.size

    # Determine the new dimensions based on the aspect ratio
    if original_width >= original_height:  # If the image is wider or square
        new_width = char_width
        new_height = int((char_width / original_width) * original_height)
    else:  # If the image is taller
        new_height = char_width
        new_width = int((char_width / original_height) * original_width)

    # Resize the image
    try:
        resized_image = char_image.resize((new_width, new_height), Image.LANCZOS)
    except:
        return None, (0, 0)

    # Convert the resized image to a numpy array
    resized_array = np.array(resized_image)

    return resized_array, (new_width, new_height)

def calculate_image_coordinates_old(char_region_x0, char_region_y0, char_region_x1, image_width, image_height):
    # 領域の幅を計算
    char_region_width = char_region_x1 - char_region_x0

    if image_width >= image_height:
        # 画像が横長の場合
        scale_factor = char_region_width / image_width
        scaled_height = int(image_height * scale_factor)
        image_x0 = char_region_x0
        image_y0 = char_region_y0
        image_x1 = char_region_x1
        image_y1 = char_region_y0 + scaled_height
    else:
        # 画像が縦長の場合
        image_y0 = char_region_y0
        image_y1 = char_region_y0 + image_height
        # scale_factor = char_region_width / image_width
        image_x0 = (char_region_x0 + char_region_x1) // 2 - image_width // 2
        image_x1 = image_x0 + image_width

    return image_x0, image_y0, image_x1, image_y1

def place_image_on_canvas(canvas, x0, y0, x1, y1, image):
    # 配置する領域の幅と高さを計算
    region_width = x1 - x0
    region_height = y1 - y0

    # 画像の幅と高さを取得
    image_height, image_width = image.shape[:2]

    # 画像のサイズが領域のサイズと異なる場合のみリサイズ
    if image_width != region_width or image_height != region_height:
        image = cv2.resize(image, (region_width, region_height), interpolation=cv2.INTER_AREA)

    # 画像をグレースケールに変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 大津の二値化で墨の部分を抽出
    _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # マスクを3チャンネルに変換
    mask_3channel = cv2.merge([mask, mask, mask])

    # 墨の部分だけを抽出（色をそのまま保持）
    ink_only = cv2.bitwise_and(image, mask_3channel)

    # キャンバスの該当領域を取得
    canvas_region = canvas[y0:y1, x0:x1]

    # キャンバスの該当領域を0にリセット（墨を置く位置をクリア）
    canvas_region = cv2.bitwise_and(canvas_region, cv2.bitwise_not(mask_3channel))

    # 墨の部分をキャンバスに加算
    canvas[y0:y1, x0:x1] = cv2.add(canvas_region, ink_only)

    return canvas
def write_csv_one_character(unicode, char_region_x0, char_region_y0, char_region_x1, char_region_y1, 
                            csv_path='./pre_training_color_annotations.csv',
                            furi_csv_path='./pre_training_color_furi_annotations.csv',
                            furi_mode=False):
    if not furi_mode:
        selected_path = csv_path
    else:
        selected_path = furi_csv_path
    with open(selected_path, 'a') as f:
        f.write(f",{unicode},{char_region_x0},{char_region_y0},{char_region_x1},{char_region_y1}")
def update_csv_one_doc(data, 
                       csv_path ,
                       furi_csv_path,
                       furi_mode=False):
    if not furi_mode:
        selected_path = csv_path
    else:
        selected_path = furi_csv_path
    with lock:
        with open(selected_path,'a') as f:
            f.write(f'\n{data[0]}')
def add_perspective_gaussian_to_canvas(canvas, points, amplitude=1.0):
    # 領域の4点を取得
    src_points = np.array(points, dtype=np.float32)

    # ガウス分布を生成するための仮想的な正方形領域を定義
    width = int(max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3])))
    height = int(max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2])))
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # ガウス分布を生成
    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(-height / 2, height / 2, height)
    x, y = np.meshgrid(x, y)
    sigma_x = width / 5.0
    sigma_y = height / 5.0
    gaussian = amplitude * np.exp(-((x**2) / (2 * sigma_x**2) + (y**2) / (2 * sigma_y**2)))

    # Perspective Transformation行列を計算
    matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    # ガウス分布をPerspective Transformationで変形
    transformed_gaussian = cv2.warpPerspective(gaussian, matrix, (canvas.shape[1], canvas.shape[0]))

    # キャンバスにガウス分布を追加
    canvas += transformed_gaussian

    return canvas
# アフィニティ計算のために、文字の対角で作られる三角形の重心を求める関数
def return_triangle_center_of_gravity(x0,y0,x1,y1):
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    point1_x = int((x0+x0+center_x) / 3)
    point1_y = int((y0+y1+center_y) / 3)
    point2_x = int((x1+x1+center_x) / 3)
    point2_y = int((y0+y1+center_y) / 3)
    return (point1_x, point1_y), (point2_x, point2_y)
def is_kanji(unicode_str):
    try:
        # Unicode文字列を実際の文字に変換
        char = chr(int(unicode_str[2:], 16))
        # Unicodeのコードポイントを取得
        code_point = ord(char)

        # ひらがな、カタカナ、数字のUnicode範囲をチェック
        if (0x3040 <= code_point <= 0x309F) or (0x30A0 <= code_point <= 0x30FF) or (0x0030 <= code_point <= 0x0039):
            return False  # ひらがな、カタカナ、数字は漢字ではない

        # 漢字のUnicode範囲をチェック
        # CJK統合漢字 (U+4E00～U+9FFF)
        # CJK統合漢字拡張A (U+3400～U+4DBF)
        # CJK統合漢字拡張B～E (U+20000～U+2FA1F)
        if (0x4E00 <= code_point <= 0x9FFF) or (0x3400 <= code_point <= 0x4DBF) or (0x20000 <= code_point <= 0x2FA1F):
            return True

        return False  # その他の文字は漢字ではない
    except (ValueError, TypeError):
        # 無効なUnicode文字列の場合はFalseを返す
        return False
def is_hiragana(unicode_str):
    """
    指定されたUnicode文字列がひらがなを表すかどうかを判定する関数。

    Args:
        unicode_str (str): Unicodeを表す文字列（例: "U+3042"）。

    Returns:
        bool: ひらがなであればTrue、それ以外はFalse。
    """
    try:
        # Unicode文字列を実際の文字に変換
        char = chr(int(unicode_str[2:], 16))
        # Unicodeのコードポイントを取得
        code_point = ord(char)

        # ひらがなのUnicode範囲をチェック (U+3040～U+309F)
        if 0x3040 <= code_point <= 0x309F:
            return True

        return False
    except (ValueError, TypeError):
        # 無効なUnicode文字列の場合はFalseを返す
        return False
    
# 文字を意図した位置に配置するための関数
def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is 3-channel
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return grayscale_image
    else:
        raise ValueError("Input image must be a 3-channel RGB numpy array.")
def invert_black_and_white(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB画像の場合
        inverted_image = 255 - image
    else:
        raise ValueError("Invalid image format. Input must be a 3-channel RGB numpy array.")
    
    return inverted_image

# 文字単体の重心を計算する関数
def calc_center_of_gravity(gray_image):
    if len(gray_image.shape) != 2:
        raise ValueError("入力は2次元のグレースケール画像である必要があります。")

    # インクの濃度（画素値）をx方向に合計
    ink_density_x = np.sum(gray_image, axis=0)

    # インクが存在しない場合は中央値を計算できない
    if np.sum(ink_density_x) == 0:
        raise ValueError("画像にインクが存在しません（画素値の合計が0です）。")

    # 各x座標をインクの濃度に応じて複製
    x_coordinates = np.arange(len(ink_density_x))
    weighted_x = np.repeat(x_coordinates, ink_density_x.astype(int))

    # 中央値を計算
    median_x = np.median(weighted_x)

    return int(median_x)
def resize_char_image(reg_width, char_image):
    # Get the original dimensions of the image
    original_width, original_height = char_image.size
    # 縦長だった時の処理
    if (original_height > original_width) and (original_height > reg_width):
        try:
            original_width = int(original_width*reg_width/original_height)
            original_height = reg_width
            if original_height <= 4 or original_width <= 4:
                return None, (0,0),0
            char_image = char_image.resize((original_width, original_height), Image.LANCZOS)
            # 縦長の場合、画像を縮小したために、縦横が小さくなりすぎる場合は飛ばす。
        except:
            pass
    # get gray image
    invert_image = invert_black_and_white(np.array(char_image))
    gray_image = convert_to_grayscale(invert_image)

    # get center of gravity
    char_left_width = calc_center_of_gravity(gray_image=gray_image)
    char_right_width = original_width - char_left_width

    distance_from_center_of_reg = int((reg_width+1)/2)

    # confirm state
    char_left_over = (char_left_width > distance_from_center_of_reg)
    char_right_over = (char_right_width > distance_from_center_of_reg)
    char_left_longer_than_right = (char_left_width >= char_right_width)
    try:
            
        if not char_left_over and not char_right_over:
            # print('type A')
            mode = 'A'
            # print(f'debug grav : {char_left_width}')
            x0 = distance_from_center_of_reg - char_left_width
            new_width = original_width
            new_height = original_height
            resized_array = np.array(char_image.resize((new_width,new_height), Image.LANCZOS))
        elif char_left_over and not char_right_over:
            # print('type B')
            mode = 'B'
            x0 = 0
            rate_old_to_new = distance_from_center_of_reg/char_left_width
            new_width = int(original_width*rate_old_to_new)
            new_height = int(original_height*rate_old_to_new)
            resized_array = np.array(char_image.resize((new_width,new_height), Image.LANCZOS))
        elif not char_left_over and char_right_over:
            # print('type C')
            mode = 'C'
            rate_old_to_new = distance_from_center_of_reg/char_right_width
            new_width = int(original_width*rate_old_to_new)
            new_height = int(original_height*rate_old_to_new)
            x0 = reg_width - new_width
            resized_array = np.array(char_image.resize((new_width,new_height), Image.LANCZOS))
        elif char_right_over and char_left_over and char_left_longer_than_right:
            # print('type D')
            mode = 'D'
            x0 = 0
            rate_old_to_new = distance_from_center_of_reg/char_left_width
            new_width = int(original_width*rate_old_to_new)
            new_height = int(original_height*rate_old_to_new)
            resized_array = np.array(char_image.resize((new_width,new_height), Image.LANCZOS))
        elif char_right_over and char_left_over and not char_left_longer_than_right:
            # print('type E')
            mode = 'E'
            rate_old_to_new = distance_from_center_of_reg/char_right_width
            new_width = int(original_width*rate_old_to_new)
            new_height = int(original_height*rate_old_to_new)
            x0 = reg_width - new_width
            resized_array = np.array(char_image.resize((new_width,new_height), Image.LANCZOS))
        else:
            print(f'error : \n{char_left_over=}\n{char_right_over=}\n{char_left_longer_than_right=}')
            return None
    except Exception as e:
        print(f'<error>\n--<state>----------\n{mode=}\n{char_left_over=}\n{char_right_over=}\n{char_left_longer_than_right=}\n--<val>----------\n{original_width=}\n{original_height=}\n{distance_from_center_of_reg=}\n{char_left_width=}\n{char_right_width=}\n{rate_old_to_new=}\n{new_width=}\n{new_height=}')
        print(e)
        # import sys
        # sys.exit()
        return None, (0, 0), 0
    return resized_array, (new_width, new_height), x0

def calculate_image_coordinates(x0, char_region_y0, image_width, image_height):
    image_x0 = x0
    image_y0 = char_region_y0
    image_x1 = x0 + image_width
    image_y1 = image_y0  + image_height
    return image_x0, image_y0, image_x1, image_y1

# 注目している行の処理をまとめて実行
def procedure_for_one_line(paper, focus_region, file_path,file_id,
                           json_gt_part, main_csv_data, furi_csv_data,
                           tow_column_flag=False, 
                           main_region_rate_ave = 70, main_region_rate_std = 5,
                           line_space_ave = 0.1, line_space_std = 3, 
                           probability_of_line_end = 0.01, 
                           tow_column_rate = 0.015,
                           furi_mode = False,
                           furi_info = True,
                           dismiss_error = True):
    upper_cood, lower_cood, left_cood, right_cood = focus_region
    # 本文とふりがなの比率を決定する
    main_region_rate = np.random.normal(loc=main_region_rate_ave, scale=main_region_rate_std, size=(1))[0]
    while main_region_rate >= 100 or main_region_rate <= 0:
        main_region_rate = np.random.normal(loc=main_region_rate_ave, scale=main_region_rate_std, size=(1))[0]
    main_char_width = int((right_cood-left_cood) * main_region_rate / 100)
    furi_char_width = int((right_cood-left_cood) * (100-main_region_rate) / 100)
    # 行間の大きさを決定する
    line_space = int(np.random.normal(loc=(right_cood-left_cood)*line_space_ave, scale=line_space_std, size=(1))[0])
    while line_space >= (right_cood-left_cood) or line_space <= 0:
        line_space = int(np.random.normal(loc=(right_cood-left_cood)*line_space_ave, scale=line_space_std, size=(1))[0])
    num_of_char = 0
    point1 = None
    point2 = None
    char_start_upper_cood = upper_cood
    char_right_cood = right_cood
    char_left_cood = left_cood
    while True:
        # この行の処理を終了し、次の行に改行する
        if np.random.rand() < probability_of_line_end:
            # print(f'skip line for random : {num_of_char=}')
            break
        # 文字を一文字ランダムに選択する
        char_image, selected_char = return_random_char(file_path, furi_mode=furi_mode)
        if char_image.size[0] <= 4 or char_image.size[1] <= 4:
            continue
        # 選択した文字のサイズを決定する
        # check
        # resized_array, (new_width, new_height) = resize_char_image(main_char_width, char_image)
        resized_array, (new_width, new_height), inner_x0 = resize_char_image(main_char_width, char_image)
        # 文字が小さすぎるとガウス分布のマップがうまく生成されないため。
        if new_width <= 4 or new_height <= 4:
            continue
        # もし選択した文字が本文の領域に収まらなかったら、breakする。
        if char_start_upper_cood + new_height >= lower_cood:
            # print(f'skip line for line end : {num_of_char=}, {char_start_upper_cood+new_height=}')
            break
        # 2段組みにするかどうかを決定する。
        if np.random.rand() < tow_column_rate and tow_column_flag == False:
            tow_column_upper = char_start_upper_cood
            tow_column_right = char_right_cood
            tow_column_middle = int((char_right_cood + char_left_cood) / 2)
            tow_column_left = char_left_cood
            tow_column_lower = lower_cood

            # print(f'start tow column right : {num_of_char=}')
            procedure_for_one_line(paper, 
                                (tow_column_upper,tow_column_lower,tow_column_left,tow_column_middle), 
                                file_path,
                                file_id=file_id,
                                json_gt_part=json_gt_part,
                                main_csv_data=main_csv_data,
                                furi_csv_data=furi_csv_data,
                                tow_column_flag=True, main_region_rate_std = 5,
                                line_space_ave = 0.1, line_space_std = 3, 
                                probability_of_line_end = 0.01,
                                furi_info=furi_info)
            # print(f'start tow column left : {num_of_char=}')
            procedure_for_one_line(paper, 
                                (tow_column_upper,tow_column_lower,tow_column_middle,tow_column_right), 
                                file_path,
                                file_id=file_id,
                                json_gt_part=json_gt_part,
                                main_csv_data=main_csv_data,
                                furi_csv_data=furi_csv_data,
                                tow_column_flag=True, main_region_rate_std = 5,
                                line_space_ave = 0.1, line_space_std = 3, 
                                probability_of_line_end = 0.01,
                                furi_info=furi_info)
            break
        # 選択した文字の配置する位置を決定する
        # check
        # image_x0, image_y0, image_x1, image_y1 = calculate_image_coordinates(
        #     char_left_cood, char_start_upper_cood, char_left_cood+main_char_width, new_width, new_height)
        image_x0, image_y0, image_x1, image_y1 = calculate_image_coordinates(char_left_cood+inner_x0, char_start_upper_cood, new_width, new_height)
        # 選択した文字を上に詰めて配置する。
        paper = place_image_on_canvas(paper, 
                                    image_x0, image_y0, image_x1, image_y1, 
                                    resized_array)
        # 配置した文字の座標と文字種類をcvsに保存する。
        if not furi_mode:
            # write_csv_one_character(selected_char, 
            #                         image_x0, image_y0, image_x1, image_y1,
            #                         csv_path=CSV_PATH)
            main_csv_data[0]+=f',{selected_char},{image_x0},{image_y0},{image_x1},{image_y1}'
            
        else:
            # write_csv_one_character(selected_char, 
            #                         image_x0, image_y0, image_x1, image_y1, 
            #                         furi_csv_path=CSV_FURI_PATH,
            #                         furi_mode=True)
            furi_csv_data[0]+=f',{selected_char},{image_x0},{image_y0},{image_x1},{image_y1}'
        # 配置した文字の領域でmain_regionを更新する
        if not furi_mode:
            # ここで、本文の領域情報をjsonに書き込み
            # json_gt['files'][file_id]['main_region'].append([image_x0, image_y0, 
            #                                                  image_x1, image_y0, 
            #                                                  image_x1, image_y1, 
            #                                                  image_x0, image_y1])
            json_gt_part['main_region'].append([image_x0, image_y0, 
                                                image_x1, image_y0, 
                                                image_x1, image_y1, 
                                                image_x0, image_y1])
            # main_region = add_perspective_gaussian_to_canvas(main_region, ((image_x0, image_y0), (image_x1, image_y0), (image_x1, image_y1), (image_x0, image_y1)), amplitude=1.0)
        else:
            # ここで、ふりがなの領域情報をjsonに書き込み
            # json_gt['files'][file_id]['furi_region'].append([image_x0, image_y0, 
            #                                                  image_x1, image_y0, 
            #                                                  image_x1, image_y1, 
            #                                                  image_x0, image_y1])
            json_gt_part['furi_region'].append([image_x0, image_y0, 
                                                image_x1, image_y0, 
                                                image_x1, image_y1, 
                                                image_x0, image_y1])
            # furi_region = add_perspective_gaussian_to_canvas(furi_region, ((image_x0, image_y0), (image_x1, image_y0), (image_x1, image_y1), (image_x0, image_y1)), amplitude=1.0)
        # もし、num_of_char != 0 なら、main_affinityを更新する。
        # 文字の対角で作られる三角形の重心を求める。
        point3, point4 = return_triangle_center_of_gravity(image_x0, image_y0, image_x1, image_y1)
        if num_of_char != 0:
            # main_affinityを更新する。
            if not furi_mode:
                # ここで本文のアフィニティをjsonに書き込み ここから
                # json_gt['files'][file_id]['main_affinity'].append([point1[0], point1[1],
                #                                                  point2[0], point2[1],
                #                                                  point4[0], point4[1],
                #                                                  point3[0], point3[1]])
                json_gt_part['main_affinity'].append([point1[0], point1[1],
                                                      point2[0], point2[1],
                                                      point4[0], point4[1],
                                                      point3[0], point3[1]])
                # main_affinity = add_perspective_gaussian_to_canvas(main_affinity, (point1, point2, point4, point3), amplitude=1.0)
            else:
                # ここでふりがなのアフィニティをjsonに書き込み
                # json_gt['files'][file_id]['furi_affinity'].append([point1[0], point1[1],
                #                                                  point2[0], point2[1],
                #                                                  point4[0], point4[1],
                #                                                  point3[0], point3[1]])
                json_gt_part['furi_affinity'].append([point1[0], point1[1],
                                                      point2[0], point2[1],
                                                      point4[0], point4[1],
                                                      point3[0], point3[1]])
                # furi_affinity = add_perspective_gaussian_to_canvas(furi_affinity, (point1, point2, point4, point3), amplitude=1.0)
        num_of_char += 1
        point1 = point3
        point2 = point4
        # もし配置した文字が漢字ならば、ふりがなを横に配置する。（ふりがな用のprocedure_for_one_lineをつくる。）
        if is_kanji(selected_char) and furi_info:
            furi_x0 = left_cood + main_char_width
            furi_y0 = image_y0
            furi_x1 = right_cood
            furi_y1 = image_y1
            procedure_for_one_line(paper, 
                                (furi_y0, furi_y1, furi_x0, furi_x1), 
                                file_path,
                                file_id=file_id,
                                json_gt_part=json_gt_part,
                                main_csv_data=main_csv_data,
                                furi_csv_data=furi_csv_data,
                                tow_column_flag=True, 
                                main_region_rate_ave = 80, main_region_rate_std = 5,
                                line_space_ave = 0.1, line_space_std = 3, 
                                probability_of_line_end = 0.01, 
                                tow_column_rate = 0.015,
                                furi_mode=True,
                                furi_info=False)
        # 次に配置する文字の位置を決定した行間を利用して、決定する。
        char_start_upper_cood = image_y1 + line_space
        char_right_cood = right_cood
        char_left_cood = left_cood
        try:
            pass
        except:
            continue
    return paper

# csvファイルにその画像の情報を追加する
def add_csv_data(file_id,
                 csv_path):
    # ファイルのID取得 str(doc_id)+'_sep_'+file_path.split('/')[-1].split('.')[0]+'.jpg'
    # CSVファイルにデータを追加
    with open(csv_path, 'a') as f:
        f.write('\n'+file_id)
def save_as_4channel_npy(array1, array2, array3, array4, file_id, save_path):
    # 4つの配列を結合して0次元方向に4チャネルの配列を作成
    # combined_array = np.stack((array1, array2, array3, array4), axis=0)

    # 保存先ディレクトリが存在しない場合は作成
    os.makedirs(save_path, exist_ok=True)

    full_path = os.path.join(save_path, file_id)

    # 配列を保存:チェック
    np.savez_compressed(full_path,main_reg=array1,main_affi = array2,furi_reg = array3,furi_affi = array4)

    return full_path
# 一枚の画像に対する処理を行う関数
def export_image_as_jpg(image: np.ndarray, 
                        output_filename: str = "output.jpg", 
                        quality: int = 95,
                        image_dir = '../kuzushiji-recognition/synthetic_images/input_images'):
    # ファイル名に拡張子が含まれていない場合、.jpgを追加
    if not output_filename.lower().endswith(".jpg"):
        output_filename += ".jpg"

    # RGB形式をBGR形式に変換
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # カレントディレクトリに保存
    # output_path = os.path.join(os.getcwd(), output_filename)
    output_path = image_dir + output_filename

    # JPEG形式で保存 (品質を指定)
    cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

    return output_path
def extract_ink_rgb_mean(image):
    """
    墨で書かれた文書の画像から墨のRGB値の平均を抽出する関数。

    Args:
        image (numpy.ndarray): 入力画像 (RGB形式)。

    Returns:
        numpy.ndarray: 墨のRGB値の平均 (1次元配列, dtype=numpy.uint8)。
    """
    # 画像をグレースケールに変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 大津の二値化で墨の部分を抽出
    _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # マスクを適用して墨の部分を抽出
    ink_pixels = image[mask == 255]

    # 墨のRGB値の平均を計算
    if len(ink_pixels) > 0:
        ink_rgb_mean = np.mean(ink_pixels, axis=0).astype(np.uint8)
    else:
        # 墨が検出されなかった場合はゼロ配列を返す
        ink_rgb_mean = np.array([0, 0, 0], dtype=np.uint8)

    return ink_rgb_mean
def draw_line(canvas, x0, y0, x1, y1, color=(255, 0, 0), thickness=2):
    """
    指定した座標に線を描画する関数。

    Args:
        canvas (numpy.ndarray): 描画先のキャンバス画像。
        x0, y0 (int): 線の始点の座標。
        x1, y1 (int): 線の終点の座標。
        color (tuple): 線の色 (BGR形式)。
        thickness (int): 線の太さ。
    """
    cv2.line(canvas, (x0, y0), (x1, y1), color, thickness)
def procedure_for_one_image(image, file_path=None,json_gt = None, show_flag=False):
    doc_id = file_path.split('/')[-3]
    file_id = str(doc_id)+'_sep_'+file_path.split('/')[-1].split('.')[0]
    # csvファイルにその画像の情報を追加する
    main_csv_data = [file_id + ',']
    furi_csv_data = [file_id + ',']
    # 変える
    # add_csv_data(file_id, csv_path=CSV_PATH)
    # add_csv_data(file_id, csv_path=CSV_FURI_PATH)
    # ground_truth Json に注目している画像の情報を追加する。
    new_json_data = {'main_region':[],
                    'main_affinity':[],
                    'furi_region':[],
                    'furi_affinity':[]}
    # 変える
    # json_gt['files'][file_id] = {'main_region':[],
    #                              'main_affinity':[],
    #                              'furi_region':[],
    #                              'furi_affinity':[]}
    # キャンバスとなるpaperを作成する。
    # チェックポイント：inpaint_level=20
    paper = remove_ink(image, show_flag=True, inpaint_radius=inpaint_level)
    
    # 文字を配置する領域を決定する。
    upper_cood, lower_cood, left_cood, right_cood= decide_upper_lower_coodinate(image)
    # 枠線を描画するかどうかを決定する。
    if np.random.rand() < 0.5:
        # 墨のRGB値の平均を取得
        ink_rgb_mean = extract_ink_rgb_mean(image)
        # 墨のRGB値の平均をBGR形式に変換
        ink_rgb_bgr = tuple(map(int, ink_rgb_mean[::-1]))
        # 枠線を描画する
        thickness = np.random.normal(loc=5, scale=1, size=(1))[0]
        while thickness >= 20 or thickness <= 0:
            thickness = np.random.normal(loc=5, scale=1, size=(1))[0]
        draw_line(paper, left_cood,upper_cood,right_cood,upper_cood, color=(ink_rgb_bgr[0],ink_rgb_bgr[1],ink_rgb_bgr[2]), thickness=int(thickness))
        draw_line(paper, left_cood,lower_cood,right_cood,lower_cood, color=(ink_rgb_bgr[0],ink_rgb_bgr[1],ink_rgb_bgr[2]), thickness=int(thickness))
        draw_line(paper, left_cood,upper_cood,left_cood,lower_cood, color=(ink_rgb_bgr[0],ink_rgb_bgr[1],ink_rgb_bgr[2]), thickness=int(thickness))
        draw_line(paper, right_cood,upper_cood,right_cood,lower_cood, color=(ink_rgb_bgr[0],ink_rgb_bgr[1],ink_rgb_bgr[2]), thickness=int(thickness))
    # 行数を決定する。
    number_of_lines = generate_uniform_integers(low=6, high=15, size=1)
    # 正解データとなるキャンバスを作成: 削除
    # main_region, main_affinity, furi_region, furi_affinity = return_ground_truth_canvas(image)
    # 文字を配置する行の座標を計算する
    focus_region_list = focus_on_one_line(upper_cood, lower_cood, left_cood, right_cood, number_of_lines)
    # ふりがなのある文書か決める
    if np.random.rand() < 0.3:
        furi_info = False
    else:
        furi_info = True
    # 一行ずつ処理を行う。
    for focus_region in focus_region_list:
        upper_cood, lower_cood, left_cood, right_cood = focus_region
        procedure_for_one_line(paper, focus_region, file_path,
                               file_id=file_id,
                               json_gt_part=new_json_data,
                               main_csv_data=main_csv_data,
                               furi_csv_data=furi_csv_data,
                               tow_column_flag=False,  main_region_rate_std = 5,
                               line_space_ave = 0.1, line_space_std = 3, 
                               probability_of_line_end = 0.01,
                               furi_info = furi_info,)
    
    # 画像をJPEG形式で保存
    export_image_as_jpg(paper, 
                        output_filename=file_id+'.jpg', 
                        quality=95,
                        image_dir=IMAGE_DIR_PATH)
    # 1枚の画像の処理が終わったらjsonを都度保存する。
    update_json_data(file_id=file_id, 
                     data=new_json_data,
                     file_path=GT_JSON_PATH)
    update_csv_one_doc(data=main_csv_data,
                       csv_path=CSV_PATH,
                       furi_csv_path= CSV_FURI_PATH,
                       furi_mode=False)
    update_csv_one_doc(data=furi_csv_data,
                       csv_path=CSV_PATH,
                       furi_csv_path= CSV_FURI_PATH,
                       furi_mode=True)
    # save_as_4channel_npy(main_region, main_affinity, furi_region, furi_affinity,
    #                      file_id=str(doc_id)+'_sep_'+file_path.split('/')[-1].split('.')[0],
    #                      save_path=GROUND_TRUTH_IMAGE_DIR_PATH)
def main_exe_for_one_image(file_path, json_gt, procedure_for_one_image=None):
    image = judge_brightness_and_return_image(file_path)
    if image is None:
        print(f"Skipping {file_path.split('/')[-1]} due to brightness.")
        return None
    else:
        procedure_for_one_image(image,file_path,json_gt)
        print(f'done : {file_path.split("/")[-1]}')
def load_GT_json(file_path):
    if not os.path.exists(file_path):
        print("json ファイルが存在しません。新しく作成します。")
        create_json_data()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("json データを読み込みました。")
    return data
def create_json_data():
    data = {
        "files":{}
    }
    save_json_data(data, GT_JSON_PATH)
    print("json 初期データを作成しました。")

# データの保存
def save_json_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("json データを保存しました。")


def update_json_data(file_id, data, file_path):
    with lock:
        json_data = load_GT_json(file_path)
        json_data['files'][file_id] = data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        with open(file_path.split('.json')[0]+'_backup.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    print("json データを保存しました。")

def main_exe(procedure_for_one_image = None, testdata_doc_id = []):
    doc_path_list = return_doc_path_list(doc_path_list=testdata_doc_id)
    json_gt = load_GT_json(GT_JSON_PATH)
    for doc_path in doc_path_list:
        file_path_list = return_file_path_list(doc_path)
        for file_path in file_path_list:
            if procedure_for_one_image != None:
                # 一枚の画像に対して、疑似的を文字を合成し、自己教師あり学習の正解データを作成する。
                main_exe_for_one_image(
                    file_path=file_path, 
                    json_gt = json_gt,
                    procedure_for_one_image=procedure_for_one_image)
# 画像全体に対する処理
inpaint_level = 20
IMAGE_DIR_PATH = '../kuzushiji-recognition/synthetic_images/input_images/'
GROUND_TRUTH_IMAGE_DIR_PATH = '../kuzushiji-recognition/synthetic_images/ground_truth_images/'
GT_JSON_PATH = '../kuzushiji-recognition/synthetic_images/gt_json.json'
CSV_PATH = '../kuzushiji-recognition/synthetic_images/pre_training_color_annotations.csv'
CSV_FURI_PATH = '../kuzushiji-recognition/synthetic_images/pre_training_color_furi_annotations.csv' 
testdata_doc_id = []
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    with Manager() as manager:
        lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = []
            results = []
            doc_path_list = return_doc_path_list(doc_path_list=testdata_doc_id)
            json_gt = load_GT_json(GT_JSON_PATH)
            for doc_path in doc_path_list:
                file_path_list = return_file_path_list(doc_path)
                for file_path in file_path_list:
                    if procedure_for_one_image != None:
                        doc_id = file_path.split('/')[-3]
                        file_id = str(doc_id)+'_sep_'+file_path.split('/')[-1].split('.')[0]
                        if file_id in json_gt['files']:
                            pass
                        else:
                            futures.append(executor.submit(
                                main_exe_for_one_image, 
                                file_path=file_path,
                                json_gt = json_gt,
                                procedure_for_one_image=procedure_for_one_image
                            ))
                            # 一枚の画像に対して、疑似的を文字を合成し、自己教師あり学習の正解データを作成する。
                            main_exe_for_one_image(
                                file_path=file_path, 
                                json_gt = json_gt,
                                procedure_for_one_image=procedure_for_one_image)
            print('Waiting for all futures to complete...')

            # for f in tqdm(as_completed(futures), total=100):
            # for f in tqdm(as_completed(futures), total=100, file=sys.stdout):
            for f in tqdm(as_completed(futures), total=100, file=sys.stdout,ncols=80, flush=True):
                results.append(f.result())
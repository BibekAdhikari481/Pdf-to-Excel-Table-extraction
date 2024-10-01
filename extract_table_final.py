from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from datetime import datetime
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize OCR and YOLO model
ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = YOLO("C:\\PDF Table Data Extraction\\Yolo_Dataset\\runs\\train\\table_model\\weights\\best (4).pt")


def convert_pdf_to_high_quality_images(pdf_path, dpi=400):
    images = convert_from_path(pdf_path, dpi)
    enhanced_images = []

    for img in images:
        img = img.convert('RGB')
        img = img.filter(ImageFilter.SHARPEN)
        enhanced_images.append(np.array(img))

    return enhanced_images


def detect_table_components(image):
    results = model.predict(source=image, imgsz=640, conf=0.2)
    return results


def extend_bboxes_to_table_boundary(bboxes, image_shape, axis):
    extended_bboxes = []
    height, width = image_shape[:2]

    for (x1, y1, x2, y2) in bboxes:
        if axis == 0:  # Columns: Extend vertically
            extended_bboxes.append((x1, 0, x2, height))
        else:  # Rows: Extend horizontally
            extended_bboxes.append((0, y1, width, y2))

    return extended_bboxes


def extract_row_column_bboxes(results, image_shape, x_tolerance=5, proximity_tolerance=30):
    """
    Extract bounding boxes for rows and columns and filter out duplicate columns.
    
    - image_shape: Shape of the image to extend bounding boxes.
    - x_tolerance: Used to check if columns overlap vertically.
    - proximity_tolerance: Used to check if columns are too close horizontally (to detect duplicates).
    """
    row_bboxes = []
    col_bboxes = []
    table_bboxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]

            if label == "Row":
                row_bboxes.append((x1, y1, x2, y2))
            elif label == "Column":
                col_bboxes.append((x1, y1, x2, y2))
            elif label == "Table":
                table_bboxes.append((x1, y1, x2, y2))

    row_bboxes = sorted(row_bboxes, key=lambda b: b[1])
    col_bboxes = sorted(col_bboxes, key=lambda b: b[0])
    
    if table_bboxes:  # Ensure there is at least one table detected
        first_table_bbox = table_bboxes[0]
        # Extend columns vertically to the table boundaries
        col_bboxes = extend_bboxes_to_table_boundary(col_bboxes, image_shape, axis=0)

    # Remove duplicate/overlapping or close columns (keep the left-most one)
    filtered_col_bboxes = []
    
    i = 0
    while i < len(col_bboxes):
        current_col = col_bboxes[i]
        current_x1 = current_col[0]

        left_most_col = current_col
        
        # Compare the current column with the next columns to see if they are within proximity
        j = i + 1
        while j < len(col_bboxes):
            next_col = col_bboxes[j]
            next_x1 = next_col[0]
            
            if abs(next_x1 - current_x1) <= proximity_tolerance:
                if next_x1 < left_most_col[0]:
                    left_most_col = next_col
                j += 1
            else:
                break
        
        filtered_col_bboxes.append(left_most_col)
        i = j

    print(f"Detected columns (before filtering): {col_bboxes}")
    print(f"Filtered columns (after removing duplicates): {filtered_col_bboxes}")
    # Extend rows horizontally to the table boundaries
    if table_bboxes:
        row_bboxes = extend_bboxes_to_table_boundary(row_bboxes, image_shape, axis=1)

    return row_bboxes, filtered_col_bboxes, table_bboxes


def extract_cells(image, row_bboxes, col_bboxes, table_bbox):
    cells = []

    if not row_bboxes or not col_bboxes or not table_bbox:
        print("Warning: No rows, columns, or table detected.")
        return cells

    for (rx1, ry1, rx2, ry2) in row_bboxes:
        row_cells = []
        for (cx1, cy1, cx2, cy2) in col_bboxes:
            # Calculate intersection of row and column for cell
            cell_x1 = max(rx1, cx1)
            cell_y1 = max(ry1, cy1)
            cell_x2 = min(rx2, cx2)
            cell_y2 = min(ry2, cy2)

            if cell_x1 < cell_x2 and cell_y1 < cell_y2:
                cell_image = image[cell_y1:cell_y2, cell_x1:cell_x2]
                cell_image_pil = Image.fromarray(cell_image)

                try:
                    result = ocr.ocr(np.array(cell_image_pil))
                    if result:
                        cell_text = ' '.join([line[1][0] for line in result[0]]).strip()
                    else:
                        cell_text = ""
                except Exception as e:
                    print(f"Error during OCR: {e}")
                    cell_text = ""

                row_cells.append(cell_text)

        cells.append(row_cells)

    '''# Normalize row lengths by padding shorter rows with empty strings
    if cells:
        max_cols = max(len(row) for row in cells)
        for row in cells:
            while len(row) < max_cols:
                row.append('')'''

    return cells


def save_to_excel(cells, output_excel_path):
    table_df = pd.DataFrame(cells)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path_with_timestamp = f"{output_excel_path}_{timestamp}.xlsx"

    try:
        table_df.to_excel(output_path_with_timestamp, index=False)
        print(f"Data successfully saved to {output_path_with_timestamp}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_pdf_to_excel(pdf_path, output_excel_path):
    images = convert_pdf_to_high_quality_images(pdf_path)
    all_cells = []

    for i, img in enumerate(images):
        img_cv = np.array(img)
        results = detect_table_components(img_cv)
        row_bboxes, col_bboxes, table_bbox = extract_row_column_bboxes(results, img_cv.shape)
        cells = extract_cells(img_cv, row_bboxes, col_bboxes, table_bbox)
        all_cells.extend(cells)

    save_to_excel(all_cells, output_excel_path)


pdf_path = "C:\\PDF Table Data Extraction\\Barclays\\Barclays Business 041.pdf"
output_excel_path = "C:\\PDF Table Data Extraction\\Barclays\\Barclays Business 041.xlsx"
process_pdf_to_excel(pdf_path, output_excel_path)

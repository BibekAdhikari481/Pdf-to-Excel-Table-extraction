'''from PIL import Image,ImageFilter
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = YOLO("C:\\PDF Table Data Extraction\\Yolo_Dataset\\runs\\train\\table_model\\weights\\best (4).pt")


def convert_pdf_to_high_quality_images(pdf_path, dpi=300):
    # Convert PDF to images at a specified DPI and apply enhancement
    images = convert_from_path(pdf_path, dpi)
    enhanced_images = []
 
    for img in images:
        # Enhance image quality (optional: convert to RGB)
        img = img.convert('RGB')

        # Apply sharpening filter
        img = img.filter(ImageFilter.SHARPEN)

        # Append the enhanced image to the list
        enhanced_images.append(np.array(img))

    return enhanced_images


def detect_table_components(image):
    results = model.predict(source=image, imgsz=640, conf=0.125)
    return results


def extract_row_column_bboxes(results):
    row_bboxes = []
    col_bboxes = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            
            if label == "Row":
                row_bboxes.append((x1, y1, x2, y2))
            elif label == "Column":
                col_bboxes.append((x1, y1, x2, y2))
    

    row_bboxes = sorted(row_bboxes, key=lambda b: b[1]) #Sort row_bboxes by the y1 coordinate (top edge) to order rows from top to bottom.
    col_bboxes = sorted(col_bboxes, key=lambda b: b[0]) #Sort col_bboxes by the x1 coordinate (left edge) to order columns from left to right.
      
    return row_bboxes, col_bboxes



def extract_cells(image, row_bboxes, col_bboxes):
    cells = []
    
    if not row_bboxes or not col_bboxes:
        print("Warning: No rows or columns detected.")
        return cells

    for (rx1, ry1, rx2, ry2) in row_bboxes:
        row_cells = []
        for (cx1, cy1, cx2, cy2) in col_bboxes:
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
        
        #if row_cells:
        cells.append(row_cells)
    
    if cells:
        print(f"Extracted {len(cells)} rows and {len(cells[0])} columns of cells.")
    else:
        print("Warning: No valid cells detected.")
    
    # Normalize row lengths by padding shorter rows with empty strings
    if cells:
        max_cols = max(len(row) for row in cells)
        for row in cells:
            while len(row) < max_cols:
                row.append('')
    
    return cells



def save_to_excel(cells, output_excel_path):
    table_df = pd.DataFrame(cells)
    print(table_df)
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
        row_bboxes, col_bboxes = extract_row_column_bboxes(results)
        cells = extract_cells(img_cv, row_bboxes, col_bboxes)
        all_cells.extend(cells)
    
    save_to_excel(all_cells, output_excel_path)



pdf_path = "C:\\PDF Table Data Extraction\\Barclays\\Statement 23-dec-22 ac 63923029 (1)-1-5.pdf"
output_excel_path = "C:\\PDF Table Data Extraction\\Barclays\\Statement 23-dec-22 ac 63923029 (1)-1-5.xlsx"
process_pdf_to_excel(pdf_path, output_excel_path)


from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = YOLO("C:\\PDF Table Data Extraction\\Yolo_Dataset\\runs\\train\\table_model\\weights\\best (4).pt")

def convert_pdf_to_high_quality_images(pdf_path, dpi=300):
    # Convert PDF to images at a specified DPI and apply enhancement
    images = convert_from_path(pdf_path, dpi)
    enhanced_images = []
 
    for img in images:
        # Enhance image quality (optional: convert to RGB)
        img = img.convert('RGB')
        # Apply sharpening filter
        img = img.filter(ImageFilter.SHARPEN)
        enhanced_images.append(np.array(img))

    return enhanced_images

def detect_table_components(image):
    results = model.predict(source=image, imgsz=640, conf=0.125)
    return results

def merge_overlapping_bboxes(bboxes):
    # Merge overlapping bounding boxes
    merged_bboxes = []
    for bbox in sorted(bboxes, key=lambda b: (b[1], b[0])):  # sort by y and then x
        if merged_bboxes and bbox[1] <= merged_bboxes[-1][3]:
            last_bbox = merged_bboxes[-1]
            merged_bboxes[-1] = (
                min(last_bbox[0], bbox[0]),  # x1
                min(last_bbox[1], bbox[1]),  # y1
                max(last_bbox[2], bbox[2]),  # x2
                max(last_bbox[3], bbox[3])   # y2
            )
        else:
            merged_bboxes.append(bbox)
    return merged_bboxes

def extract_row_column_bboxes(results):
    row_bboxes = []
    col_bboxes = []
    table_bbox = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            
            if label == "Row":
                row_bboxes.append((x1, y1, x2, y2))
            elif label == "Column":
                col_bboxes.append((x1, y1, x2, y2))
            elif label == "Table":
                table_bbox = (x1, y1, x2, y2)  # Store the table boundary

    # Merge overlapping rows and columns
    row_bboxes = merge_overlapping_bboxes(sorted(row_bboxes, key=lambda b: b[1]))
    col_bboxes = merge_overlapping_bboxes(sorted(col_bboxes, key=lambda b: b[0]))
    
    return row_bboxes, col_bboxes, table_bbox

def extend_to_table_border(bbox, table_bbox, axis='horizontal'):
    # Extend row/column to the table's boundary if needed
    x1, y1, x2, y2 = bbox
    tb_x1, tb_y1, tb_x2, tb_y2 = table_bbox

    if axis == 'horizontal':
        x1 = tb_x1 if x1 < tb_x1 else x1
        x2 = tb_x2 if x2 > tb_x2 else x2
    elif axis == 'vertical':
        y1 = tb_y1 if y1 < tb_y1 else y1
        y2 = tb_y2 if y2 > tb_y2 else y2

    return (x1, y1, x2, y2)

def extract_cells(image, row_bboxes, col_bboxes, table_bbox):
    cells = []

    if not row_bboxes or not col_bboxes or not table_bbox:
        print("Warning: No rows, columns, or table detected.")
        return cells

    for (rx1, ry1, rx2, ry2) in row_bboxes:
        row_cells = []
        extended_row_bbox = extend_to_table_border((rx1, ry1, rx2, ry2), table_bbox, axis='vertical')

        for (cx1, cy1, cx2, cy2) in col_bboxes:
            extended_col_bbox = extend_to_table_border((cx1, cy1, cx2, cy2), table_bbox, axis='horizontal')
            
            cell_x1 = max(extended_row_bbox[0], extended_col_bbox[0])
            cell_y1 = max(extended_row_bbox[1], extended_col_bbox[1])
            cell_x2 = min(extended_row_bbox[2], extended_col_bbox[2])
            cell_y2 = min(extended_row_bbox[3], extended_col_bbox[3])

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
        row_bboxes, col_bboxes, table_bbox = extract_row_column_bboxes(results)
        cells = extract_cells(img_cv, row_bboxes, col_bboxes, table_bbox)
        all_cells.extend(cells)
    
    save_to_excel(all_cells, output_excel_path)


pdf_path = "C:\\PDF Table Data Extraction\\Barclays\\Image (2.1-60-70.pdf"
output_excel_path = "C:\\PDF Table Data Extraction\\Barclays\\Image (2.1-60-70.xlsx"
process_pdf_to_excel(pdf_path, output_excel_path)'''



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
model = YOLO("C:\\PDF Table Data Extraction\\Yolo_Dataset\\runs\\train\\table_model\\weights\\best (5).pt")


def convert_pdf_to_high_quality_images(pdf_path, dpi=300):
    # Convert PDF to images at a specified DPI and apply enhancement
    images = convert_from_path(pdf_path, dpi)
    enhanced_images = []

    for img in images:
        img = img.convert('RGB')
        img = img.filter(ImageFilter.SHARPEN)
        enhanced_images.append(np.array(img))

    return enhanced_images


def detect_table_components(image):
    # Run YOLO model to detect table components
    results = model.predict(source=image, imgsz=640, conf=0.2)
    return results


def extend_row_to_table_border(row_bbox, table_bbox):
    """ Extend row bounding box to table borders horizontally. """
    rx1, ry1, rx2, ry2 = row_bbox
    tx1, ty1, tx2, ty2 = table_bbox
    # Extend row left and right boundaries to table
    return (tx1, ry1, tx2, ry2)


def extend_column_to_table_border(col_bbox, table_bbox):
    """ Extend column bounding box to table borders vertically. """
    cx1, cy1, cx2, cy2 = col_bbox
    tx1, ty1, tx2, ty2 = table_bbox
    # Extend column top and bottom boundaries to table
    return (cx1, ty1, cx2, ty2)


def extract_row_column_bboxes(results, x_tolerance=5, proximity_tolerance=30): 
    """
    Extract bounding boxes for rows and columns and filter out duplicate columns.
    
    - x_tolerance: Used to check if columns overlap vertically.
    - proximity_tolerance: Used to check if columns are too close horizontally (to detect duplicates).
    """
    row_bboxes = []
    col_bboxes = []
    table_bbox = None  # To store table bounding box

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]

            if label == "Row":
                row_bboxes.append((x1, y1, x2, y2))
            elif label == "Column":
                col_bboxes.append((x1, y1, x2, y2))
            elif label == "Table":
                table_bbox = (x1, y1, x2, y2)  # Store the table bounding box

    # Sort rows by top y-coordinate and columns by left x-coordinate
    row_bboxes = sorted(row_bboxes, key=lambda b: b[1])
    col_bboxes = sorted(col_bboxes, key=lambda b: b[0])

    # Step 1: Extend each column to the top and bottom of the table
    col_bboxes = [extend_column_to_table_border(col_bbox, table_bbox) for col_bbox in col_bboxes]

    # Step 2: Remove duplicate/overlapping or close columns (keep the left-most one)
    filtered_col_bboxes = []
    
    i = 0
    while i < len(col_bboxes):
        # Start with the current column
        current_col = col_bboxes[i]
        current_x1 = current_col[0]
        
        # Initialize a variable to hold the left-most column in a group
        left_most_col = current_col
        
        # Compare the current column with the next columns to see if they are within proximity
        j = i + 1
        while j < len(col_bboxes):
            next_col = col_bboxes[j]
            next_x1 = next_col[0]
            
            # If the next column is within proximity, we consider it as a duplicate candidate
            if abs(next_x1 - current_x1) <= proximity_tolerance:
                # Keep the left-most column (with the smallest x1 value)
                if next_x1 < left_most_col[0]:
                    left_most_col = next_col
                j += 1
            else:
                # Break the loop if the next column is not in proximity
                break
        
        # Add the left-most column to the filtered list
        filtered_col_bboxes.append(left_most_col)
        
        # Skip all columns we just compared
        i = j

    # Print columns detected before filtering
    print(f"Detected columns (before filtering): {col_bboxes}")

    # Print filtered columns after removing duplicates
    print(f"Filtered columns (after removing duplicates): {filtered_col_bboxes}")


    return row_bboxes, filtered_col_bboxes, table_bbox
 


def extract_cells(image, row_bboxes, col_bboxes, table_bbox):
    cells = []

    if not row_bboxes or not col_bboxes or not table_bbox:
        print("Warning: No rows, columns, or table detected.")
        return cells

    # Extend each row horizontally to match table borders
    row_bboxes = [extend_row_to_table_border(row_bbox, table_bbox) for row_bbox in row_bboxes]

    # Extend each column vertically to match table borders
    col_bboxes = [extend_column_to_table_border(col_bbox, table_bbox) for col_bbox in col_bboxes]

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

    # Normalize row lengths by padding shorter rows with empty strings
    if cells:
        max_cols = max(len(row) for row in cells)
        for row in cells:
            while len(row) < max_cols:
                row.append('')

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
        row_bboxes, col_bboxes, table_bbox = extract_row_column_bboxes(results)
        cells = extract_cells(img_cv, row_bboxes, col_bboxes, table_bbox)
        all_cells.extend(cells)

    save_to_excel(all_cells, output_excel_path)


# Example usage
pdf_path = "C:\\PDF Table Data Extraction\\Barclays\\Statement 23-dec-22 ac 63923029 (1)-1-5.pdf"
output_excel_path = "C:\\PDF Table Data Extraction\\Barclays\\Statement 23-dec-22 ac 63923029 (1)-1-5.xlsx"
process_pdf_to_excel(pdf_path, output_excel_path)




 

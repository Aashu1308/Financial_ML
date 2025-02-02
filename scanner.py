import cv2
import pytesseract
import re
import numpy as np


def extract_payment_info(image_path):
    print("\n[INFO] Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Failed to load image. Check the file path.")
        return None, None

    print("[INFO] Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance image preprocessing
    print("[INFO] Applying image preprocessing...")
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    print("[INFO] Running OCR...")
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    if not data or "text" not in data:
        print("[ERROR] OCR returned no text data.")
        return None, None

    print("[INFO] Extracting text...")
    extracted_text = data["text"]
    print("[DEBUG] Raw OCR output:", extracted_text)

    found_total = None
    found_visa = None
    total_coords = None
    visa_coords = None

    print("[INFO] Searching for amounts in extracted text...")

    # Search for total amounts in a window-based approach
    for i in range(len(extracted_text)):
        current_text = extracted_text[i].strip().upper()

        # Check for keywords indicating totals
        if current_text in ['TOTAL', 'GRAND TOTAL']:
            # Look in a window of 3 positions before and after for amounts
            window_start = max(0, i - 3)
            window_end = min(len(extracted_text), i + 4)

            for j in range(window_start, window_end):
                text = extracted_text[j].strip()
                # Match numbers with optional decimals and commas
                if re.match(r'^[\d,.]+$', text):
                    try:
                        # Verify it's a valid number
                        test_amount = float(text.replace(',', ''))
                        if test_amount > 0:  # Only accept positive amounts
                            if current_text == 'GRAND TOTAL':  # Prioritize grand total
                                found_total = text
                                total_coords = (
                                    data["left"][i],
                                    data["top"][i],
                                    data["width"][i],
                                    data["height"][i],
                                )
                                print(f"[SUCCESS] Grand total found: {found_total}")
                                break
                            elif (
                                not found_total
                            ):  # Use regular total if grand total not found yet
                                found_total = text
                                total_coords = (
                                    data["left"][i],
                                    data["top"][i],
                                    data["width"][i],
                                    data["height"][i],
                                )
                                print(f"[SUCCESS] Total found: {found_total}")
                    except ValueError:
                        continue

        # Check for VISA payments
        if current_text == 'VISA':
            # Look ahead for amounts
            for j in range(i + 1, min(i + 4, len(extracted_text))):
                text = extracted_text[j].strip()
                if re.match(r'^[\d,.]+$', text):
                    found_visa = text
                    visa_coords = (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                    print(f"[SUCCESS] Visa payment found: {found_visa}")
                    break

    # Draw bounding boxes and labels
    if total_coords:
        x, y, w, h = total_coords
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Total: {found_total}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if visa_coords:
        x, y, w, h = visa_coords
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"Visa: {found_visa}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Save the result
    output_path = "highlighted_bill.jpg"
    print(f"[INFO] Saving highlighted image to {output_path}...")
    cv2.imwrite(output_path, img)

    # Convert amounts to float if found
    total_amount = float(found_total.replace(',', '')) if found_total else None
    visa_amount = float(found_visa.replace(',', '')) if found_visa else None

    return total_amount, visa_amount


# Example usage
if __name__ == "__main__":
    total, visa = extract_payment_info("bills/food.png")
    print(f"\nResults:")
    print(f"Total Amount: {total}")
    print(f"Visa Payment: {visa}")

import cv2

def enhance_lighting(frame):
    """
    Applies CLAHE lighting enhancement to improve recognition
    in low light, shadow, mask/glasses conditions.
    """
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except:
        # In case of conversion error, return original
        return frame

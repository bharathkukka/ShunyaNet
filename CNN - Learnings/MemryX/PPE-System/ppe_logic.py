# ppe_logic.py
from typing import List, Tuple, Dict
import time

# config: map class ids to names
CLASS_PERSON = 0
CLASS_HARDHAT = 1
CLASS_VEST = 2

def group_detections(dets: List[Tuple[float, float, float, float, float, int]]):
    """
    Input: list of (x1,y1,x2,y2,score,class_id)
    Return: per-person dict with nearby PPE detections
    """
    persons = []
    ppe = []
    for (x1,y1,x2,y2,score,cls) in dets:
        if int(cls) == CLASS_PERSON:
            persons.append((x1,y1,x2,y2,score,cls))
        else:
            ppe.append((x1,y1,x2,y2,score,cls))

    # for each person, check if PPE bboxes overlap or are within proximity
    results = []
    for px1,py1,px2,py2,ps,pc in persons:
        has_hardhat = False
        has_vest = False
        for bx1,by1,bx2,by2,bs,bcls in ppe:
            # compute IoU or simple center-in-box test
            cx = (bx1+bx2)/2
            cy = (by1+by2)/2
            # treat hardhat expected above head: y around top region
            if (cx >= px1 and cx <= px2 and cy >= py1 and cy <= py2):
                if int(bcls) == CLASS_HARDHAT and bs>0.3:
                    has_hardhat = True
                if int(bcls) == CLASS_VEST and bs>0.3:
                    has_vest = True
        compliant = has_hardhat and has_vest
        results.append({
            "person_bbox": (px1,py1,px2,py2),
            "has_hardhat": has_hardhat,
            "has_vest": has_vest,
            "compliant": compliant
        })
    return results

def any_violation(p_results):
    "Return True if any person is non-compliant"
    for r in p_results:
        if not r["compliant"]:
            return True
    return False

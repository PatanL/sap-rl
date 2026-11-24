#!/bin/bash
# usage: view.sh <frame-name>  -> writes half-size copy to scratchpad cur.jpg
S=/tmp/claude-1000/-home-dzliu-pzliu/f7c77898-47a1-451e-a609-998d61e2f9e2/scratchpad/v
/home/dzliu/pzliu/.venv/bin/python -c "
import cv2,sys
im=cv2.imread('/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/session2/frames/$1.jpg');cv2.imwrite('$S/cur.jpg',cv2.resize(im,(960,540)))"

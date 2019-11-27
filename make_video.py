import cv2
import glob


out = cv2.VideoWriter('assignment.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (1024, 934))
for filename in sorted(glob.glob('res/*.png')):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    out.write(img)
out.release()

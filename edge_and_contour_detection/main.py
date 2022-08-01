import cv2
image = cv2.imread("HandGesture.png")

#image preprocessing

#gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#threshold
threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 5)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("gray version", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("after threshold", threshold_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# now creating Edges


edged = cv2.Canny(gray, 30, 200)
cv2.imshow("edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Creating Contours

contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.waitKey(0)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
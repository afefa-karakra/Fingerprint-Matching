import os 
import cv2 
import numpy as np
import matplotlib.pyplot as plt

sample = cv2.imread("archive/SOCOFing/Altered/Altered-Medium/150__M_Right_index_finger_Obl.BMP")
threshold = 0.4
best_score = 0
false_match_rates = []
false_non_match_rates = []
genuine_matches = 0
imposter_matches = 0
total_genuine = 0
total_imposter = 0

for file in os.listdir("archive/SOCOFing/Real")[:1000]:
    fingerprint_image = cv2.imread("archive/SOCOFing/Real/" + file)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
    distances = []  # Store distances between keypoints

    for p, q in matches:
        if p.distance < threshold * q.distance:
            match_points.append(p)
            distances.append(p.distance)

        if len(match_points) > 0:
        # Genuine match
            total_genuine += 1
        if distances and min(distances) <= threshold:  # Check if distances is not empty
            genuine_matches += 1
            print("Match")
        else:
            print("Not Match")
    else:
        # Imposter match
        total_imposter += 1
        if distances and max(distances) > threshold:  # Corrected condition
            imposter_matches += 1
            print("Match")
        else:
            print("Not Match")




    keypoints = min(len(keypoints_1), len(keypoints_2))
    match_percent = len(match_points) / keypoints * 100

    if match_percent > best_score:
        best_score = match_percent
        print('The best match :', file)
        print('The score :', best_score)
        result = cv2.drawMatches(sample, keypoints_1, fingerprint_image, keypoints_2, match_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result = cv2.resize(result, None, fx=2.5, fy=2.5)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fnmr = genuine_matches / total_genuine
        fmr = imposter_matches / total_imposter
        false_non_match_rates.append(fnmr)
        false_match_rates.append(fmr)
        print("fnmr:", fnmr)
        print("fmr:", fmr)

        

        # Plot ROC curve
        plt.plot(false_match_rates, false_non_match_rates)
        plt.xlabel('False Match Rate (FMR)')
        plt.ylabel('False Non-Match Rate (FNMR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True)
        plt.show()

        # Iterate over the false match rates and false non-match rates
for i in range(len(false_match_rates)):
    if false_match_rates[i] >= false_non_match_rates[i]:
        eer_fmr = false_match_rates[i]
        eer_fnmr = false_non_match_rates[i]
        break

# Determine the threshold at which EER occurs
eer_threshold_index = np.argmin(np.abs(np.array(false_match_rates) - eer_fmr))
eer_threshold = false_match_rates[eer_threshold_index]

if (eer_fmr == eer_fnmr ):
    print("Equal Error Rate (EER):", eer_threshold)



print("Corresponding FMR and FNMR:", eer_fmr, eer_fnmr)


rm -rf croppedImages
mkdir croppedImages
dirs=(YFT DOL LAG SHARK BET ALB)
for dir in "${dirs[@]}"
do
    mkdir "croppedImages/$dir"
done
python cropImages.py

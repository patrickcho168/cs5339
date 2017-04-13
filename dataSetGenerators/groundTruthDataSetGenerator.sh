rm -rf ../croppedImagesGroundTruth
mkdir ../croppedImagesGroundTruth
dirs=(YFT DOL LAG SHARK BET ALB NoF OTHER)
for dir in "${dirs[@]}"
do
    mkdir "../croppedImagesGroundTruth/$dir"
done
python groundTruthDataSetGenerators.py
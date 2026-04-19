#!/bin/sh

urls="http://human.brain-map.org/api/v2/well_known_file_download/178238387
http://human.brain-map.org/api/v2/well_known_file_download/178238373
http://human.brain-map.org/api/v2/well_known_file_download/178238359
http://human.brain-map.org/api/v2/well_known_file_download/178238316
http://human.brain-map.org/api/v2/well_known_file_download/178238266
http://human.brain-map.org/api/v2/well_known_file_download/178236545"

mkdir -p ./data/raw/allen_HBA     && \
cd ./data/raw/allen_HBA     && \

start=$(date +"%T")
echo "Start time : $start"
# start counter
SECONDS=0

for url in $urls; do
   echo "fetching $url"
   curl -O -J $url &
done

wait  #wait for all background jobs to terminate

for i in *.zip; do
    unzip "$i" -d "${i%%.zip}";
done

rm *.zip && \

fetal_urls="http://www.brainspan.org/api/v2/well_known_file_download/278442900
http://www.brainspan.org/api/v2/well_known_file_download/278444085
http://www.brainspan.org/api/v2/well_known_file_download/278444090
http://www.brainspan.org/api/v2/well_known_file_download/278444094"

cd ..     && \
mkdir -p ./allen_human_fetal_brain      &&\
cd ./allen_human_fetal_brain

for url in $fetal_urls; do
   echo "fetching $url"
   curl -O -J $url &
done

wait  #wait for all background jobs to terminate

for i in *.zip; do
    unzip "$i" -d "${i%%.zip}";
done
rm *.zip && \


endtime=$(date +"%T")
echo "End time : $endtime"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

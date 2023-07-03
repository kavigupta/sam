#!/bin/bash

31 c -n synth_%s_%te_%tu_%nb -wn synthetic \
    -f %s 1,2,3,4,5,6,7,8,9,10 \
    -f %te 5,10,20 \
    -f %tu 5,20 \
    -f %nb 0,0.1,0.25,0.75 \
    -w 100 -fw %w $(echo {1..100} | sed 's/ /,/g') \
    "echo quiet; taskset -c %w nice -n 19 python -m working.synthetic_data.experiment main --seed %s --temperature %te --temperature-mut %tu --negativity-bias %nb"

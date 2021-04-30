NUM_FILES=`wc -l $1 | awk '{print $1}' `
split -l $((${NUM_FILES} / $2)) $1


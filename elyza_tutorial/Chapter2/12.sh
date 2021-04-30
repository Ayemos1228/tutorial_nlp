awk '{print $1}' popular-names.txt > col1.txt
awk '{print $2}' popular-names.txt > col2.txt

# cut -f 1 popular-names.txt
# cut -f 2 popular-names.txt

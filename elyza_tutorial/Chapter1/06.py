str1 = "paraparaparadise"
str2 = "paragraph"

def n_gram(seq, n):
	ans = []
	for i in range (len(seq) - n + 1):
		tmp = ()
		for j in range (n):
			tmp = tmp + tuple(seq[i + j])
		ans.append(tmp)
	return (ans)

X = set(n_gram(str1, 2))
Y = set(n_gram(str2, 2))

print(f"intersection: {X & Y}")
print(f"union: {X | Y}")
print(f"Difference: {X - Y}")
print("se in X:", {('s', 'e')} <= X)
print("se in X:", {('s', 'e')} <= Y)


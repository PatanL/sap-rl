"""Locate a resting stat pair on the ten visible battle slots; return null off-grid."""
def physical(x):
 left=x<960;v=(x-(113 if left else 1087))/179.7
 k=round(v)
 if not 0<=k<=4 or abs((v-k)*179.7)>32:return None
 return 4-k if left else k

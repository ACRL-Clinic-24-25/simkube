[
range(0; length) as $i |
[ .[] | . ] |
.[$i].spec.replicas |= . + 1
] 
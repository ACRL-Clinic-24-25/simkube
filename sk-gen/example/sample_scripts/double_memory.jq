[
range(0; length) as $i |
[ .[] | . ] |
.[$i].spec.template.spec.containers[0].resources.requests.memory |=
    (capture("(?<num>^[0-9]+)(?<unit>.*)") | "\((.num | tonumber) * 2)\(.unit)")
] 
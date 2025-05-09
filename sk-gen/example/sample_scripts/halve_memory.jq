[
range(0; length) as $i |
([ .[] | . ] |
.[$i].spec.template.spec.containers[0].resources.requests.memory |=
    (capture("(?<num>^[0-9]+)(?<unit>.*)")
    | ((.num | tonumber) / 2) as $halved_value
    | "\(($halved_value | floor))\(.unit)")
) as $modified_list
|
($modified_list[$i].spec.template.spec.containers[0].resources.requests.memory
    | capture("(?<num>^[0-9.]+)(?<unit>.*)")
    | (.num | tonumber) as $num
    | .unit as $unit
    | if ($unit == "Gi" and ($num * 1024) < 256) or ($unit == "Mi" and $num < 256)
    then empty else $modified_list end
)
] 
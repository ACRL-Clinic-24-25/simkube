[
range(0; length) as $i |
try (
    [ .[] | . ] |
    if .[$i].spec.replicas > 1
    then .[$i].spec.replicas |= . - 1
    else error("Cannot decrement replica count")
    end
) catch empty
] 
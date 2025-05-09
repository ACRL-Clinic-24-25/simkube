[
range(0; length) as $i |
try (
    [ .[] | . ] |
    .[$i].spec.template.spec.containers[0].resources.requests.cpu |= (
    capture("(?<num>^[0-9]+(?:\\.[0-9]+)?)(?<unit>m?)") |
    (
        (if .unit == "m"
        then (.num | tonumber)
        else (.num | tonumber * 1000)
        end * 2) as $mcores

        | if ($mcores % 1000 == 0)
        then "\($mcores / 1000)"
        else "\($mcores)m"
        end
    )
    )
) catch empty
] 
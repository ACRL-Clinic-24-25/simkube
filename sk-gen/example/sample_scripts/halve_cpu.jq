[
range(0; length) as $i |
[ .[] | . ] |
try (
    .[$i].spec.template.spec.containers[0].resources.requests.cpu |= (
    capture("(?<num>^[0-9]+(?:\\.[0-9]+)?)(?<unit>m?)") |
    (
        (if .unit == "m"
        then (.num | tonumber)
        else (.num | tonumber * 1000)
        end) as $mcores

        | if ($mcores / 2) >= 1
        then
            ($mcores / 2) as $new_mcores
            | if .unit == "m"
            then "\($new_mcores | floor)m"
            else "\($new_mcores / 1000)"
            end
        else error("Cannot decrement CPU")
        end
    )
    )
) catch .
] 
# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-154227e12deeada99188001de1f98c7859b44b0240a0a63280198f0600727836`
- fixture hash: `sha256-141e98c67b76e6b544c136b2dc9ec311316dae947241f48af13f9b3f509e9c48`
- score hash: `sha256-4e13af746fc06adbc43b2db5c5b318b606f6427337bb272db773a73f544a6668`
- bundle hash: `sha256-819f3a152bceaaad4dfc9005dc8399477e87a70f01f71c5373a3245f155a18bc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fd20c45ec549a50541ad825ca2263c2905bab11bd8f991e3eba1789bd6eddad |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-04eae37821fd01c33ea48228d02f215cb0a3b40129f4781a838658d4fa125813 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ab51ce64700f77ea32903ffe87f9219324d2d5259bc462b892651b4e34e6e5da |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7cf99b93ff4bd96923482dfd199154b0acef201ea2e6d7c3844c4ef55af96b7a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9a0847a9 | sha256-2bf05b496bb97b6c84303a5ae0abf1236a7baee060d01d931f82c3c14a22dfed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9a0847a9 | sha256-33d5ae9803ce016ba8fb2439c38a72c2d13667cc4b4ac7377dfe4d2e44717fd5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-003758d0 | sha256-51c7aec266a9a49276402e201e2fcceaab86fd38dda7e37dffbc31797f643d29 |

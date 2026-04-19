# Recorded Session Replay Proof Bundle

- trace id: `live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-44fbcb6576f6f004911009e236161c3fb072626b9bf71fadcefa3c9dfc1347dc`
- fixture hash: `sha256-178059882cfa4f40ce27919272b11654587c109af203796638882d20de0899c6`
- score hash: `sha256-e432224057b64804123430f6a14100a5216352c63e837a25a1cc52b699000c74`
- bundle hash: `sha256-c20da61079d8b45c8ab11b20389fec1234478a567ad8fa440a835c9b0c29f144`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc575aa7f7247c73c93f72af53d1bbeba87c049e551196e9ed2534df2a742d2 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-ba713d514b74a17a554ef76c20a988bee5415db44afb3b18f3642c680eb899ff |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-19f82afca23be85e7abc53eff2c3fe0d4b8eb79b5be5504c56a8da769d410ce2 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-cb9edd61b76529ca9e2eca7175000b5caf8c5a754b7e06e983a26650bda22d0c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-9eade00b | sha256-12250ee58496901d81cea5933753e276284ba1c5077cac1b1391f4f24c8a9fc1 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-9eade00b | sha256-b7161925f5d43314d3a78ace17ac10f828b1c5f136cb6aa39643d9863441f596 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-9eade00b | sha256-f4adb286d213c9555540be3a512e07dabbdfec186073906eb88c4008935477e7 |

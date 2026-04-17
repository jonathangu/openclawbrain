# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-529373cf8f7314054ee5a9938b5133a303e70a9153c03b373cae4ff852f394c7`
- fixture hash: `sha256-c16690eb3752325552dd8dd957f6a57c852c3d697d1ce7463c9556556d92ca19`
- score hash: `sha256-17f0a55f563d61f8f1fd1f95b7c6fa0a3c526f449c73c8facd1f01e1db230bba`
- bundle hash: `sha256-08cbc960072745d7c653875e4e806f6707f894d590b51fd7fc8b10cd8a5deda3`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9641da18215ca2d07fc313a19aa471e30d85d3a5754d470ceff969f5080d786d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-26a8bf4002c379e5108f04c35cdd0f267b50e57c46b18b353d191bd05d39f19b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-88e16cfa3ea29e182c1c64fccf1cdd5e5a6a3597b381e51c7115da066cd89619 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-00ab87acb1ab7820317bcd5baa148ab1c65b2e50e34d660a89c121f141858ec5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-46174b37 | sha256-3e4a68cf21751ac9238e2b7391a6148fe37c1652d4548b254cdd87cdd3141650 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-46174b37 | sha256-91588ab86723afe5a09669951daa546c21a56be64b0f8d909faa96056ffed571 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f7d6dba2 | sha256-4c77b2fe25e573bbcc74ad6c97babd92a051c363dfd1b15c98c52b942742dff2 |

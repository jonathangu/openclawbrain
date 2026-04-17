# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3d634932f1bce6d4693c067f779badfb747407b8de4f4dc108015f5fd2e78b`
- fixture hash: `sha256-178c9c975a3f9bee04b778ee3424e4eb908e1106cd7f867502edc61b1de425cc`
- score hash: `sha256-b92d4942a3eec0549235bc15fd01da705d0c076b9232974e1984a4638145286c`
- bundle hash: `sha256-c73e5edf0316a1eb4cf3129453125b429abfdeb7901f4b2d429265eb352cbad2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-252786d108eae05056482d31aaad41cb1fd7abe9a8bca72a4a7a00c78ba84b59 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1fae271a3685548db681777d2922711dc64ef014705b4c9c2eb4bfac4e5d12f0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d098285aea85112f6abfa8c0269060ab7a1054d21b4934051d238fe8caefffa1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-95b6a1b9cb75e5121a976ee6e0fcd7c9d3ce8f6b0fda5db65a60d11b20d3dd77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ac99f307 | sha256-047aad4513daa70db860bf661ddc82bbd0aca476c5439c18974fd1182af6a72c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ac99f307 | sha256-84db0f1f501965c6737bb4faebb03fefcfbb22ada621c298a09503fb4cf8fdfe |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-842739e6 | sha256-f62f1eb8680a899681f42464546447b566ab508fa1ba06fa7919b25d1a0559a9 |

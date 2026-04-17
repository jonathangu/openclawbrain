# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f8bd6e98ba409d4b92ff33d315c90158dc9f7928f49ee95918b29862594fc07`
- fixture hash: `sha256-f2d0f492e33718dcda5e95309dd8b8ae83d2a012ce623b86565c773255e59638`
- score hash: `sha256-2ef20041e2b276ee48254b8caa8185b3be47d856e9cedaeaf19895070686d5fe`
- bundle hash: `sha256-1c9d0d7164bfd0f2929f8793f321de3157cb0c53558b7c5ec2aad6c81e165adf`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3a49035e9fd3e0717342039595aabed753c46d3f982a6fbdc847832f0114d10f |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-36f9037b406a5475438807aad98b0dfb98a99415eeefdfc48c65e0f00bf11339 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-48af7f98890ea127ba3c328249c9a0af2f2bfd6278ccdebcc756b15400e7722e |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3c8ef95f1db9ae456a92656fa93f57c7ba5c7b84497d2e628ca24461193a15fa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5593370a | sha256-0b28ab5c662aa023376df908620b6404d2d99a2ee5fd061bbb0768d78c65addc |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5593370a | sha256-acd8340e450e87dfe52d06d4f1ce742990b517c3a3e6c59445f8f9be5d041faf |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-00eeb29b | sha256-1ab0aa66abf198faf7d3fbe2ab7b80d5e88446d848814481838f33ca8b1b8736 |

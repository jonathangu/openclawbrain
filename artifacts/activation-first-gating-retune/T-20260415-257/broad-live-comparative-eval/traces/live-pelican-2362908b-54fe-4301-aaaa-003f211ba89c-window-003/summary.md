# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69683abc74146be49e8afbd73d2f629322351b8f1ff326bedad7089f23b35cc`
- fixture hash: `sha256-78ae89352ee0e2620fdc9e4b5d6b74ee70bb4cf28775ccac9315ef7f4b6b2525`
- score hash: `sha256-02c56ab488eed5d40776a44c401bcf77d24de58f181bac4c6bd36569c2230e0f`
- bundle hash: `sha256-3f1a4bb812715a82c77ed0168b0d471fd76f6fe7c9cd056b7a831e27941d1166`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81a98d75515ca1c6519d32d4f8b5120338f9765022c93b90e0504e9561ef38af |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-613457d47659f87605a2112c32ad0cbb07b2a936b3b44b33d4b486c9e877e497 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d612b4a15aab740cdd0920e72f99ccf9c45f224a88d804e9ade34e1107dbb77d |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-881f358e38e2feac833ef2f1dfab5761dfb2848d9c6fb9ba858c2117777d9c12 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6d1ab625 | sha256-bd624ae7f70f2889291d04a6bb60516491b5d43c55966771c6d991dae3c210da |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6d1ab625 | sha256-b7fdfbb1b4bba3acf5f1098600a3fed691929206660638f02fb6402b6f5a10ec |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a36dec12 | sha256-6fa9781f6102b90b22e86f2dff818ab04db2834618068313de42e0e061ae5612 |

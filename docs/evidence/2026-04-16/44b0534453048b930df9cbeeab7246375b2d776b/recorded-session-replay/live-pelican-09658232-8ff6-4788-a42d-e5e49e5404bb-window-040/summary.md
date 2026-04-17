# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7095a4d9ce26969c4dde9c329e749be730ceb1f708c47df4f4c59a5abea7434f`
- fixture hash: `sha256-107b047d2badf45fec45fded8a1234ee55c336b1a2803fdeba6955f2f30cad1f`
- score hash: `sha256-913a60ebd9c2a0e3a296d66b277579b6467731044d81cc789300bb24caba2fec`
- bundle hash: `sha256-287ba893a523575b2a26ef641e9bb7230e0e665569afcdd81179b280e2c720ad`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8413d38761902f8b7b6bde87782ba48c8aa416069cad02d85c57f922d6bd4f24 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-886e361310f34d6213b1188e8edd09b42b89d2d462d1175c17095ea88f61decd |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7206784a2b682a298453c87070d2b6e07e4972e50a2707968620f04e14782b2a |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-9decb5eba1fe09dcc997789b390d48753356452fdc8e3882c809a633d3c867f1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-42e8d24f | sha256-13ed2333b74091f1b43f3c236428e834594162f8b8e2138c58223f40d5691aa6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-42e8d24f | sha256-903c1c50ed547dfc5d5999673d770cd917ee8221d84a063a220ea64e25478f33 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-f4563d2a | sha256-2d49e96d518180ca90a72efef6e9aeec8ba3f37b7ed77b56d195c1a79429d021 |

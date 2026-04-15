# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-82b94292f904129190996d09645352442519cd34f4a6fe4ddc3d8ccfdc15ed4f`
- fixture hash: `sha256-2b7971a9291be722d620678727dc2afe570e5b9dc9a97d0983cbb8375a8b4f0f`
- score hash: `sha256-5e6161ecf876003760ed2809a22c16652b0d743ccd556556f7045858555f90f3`
- bundle hash: `sha256-beeac02e78962ace191657b41fd65384823afb7b8745b28411a31950398777f8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2126389181abd46124f339c97d016b2e80dbdd1c3f4a30cb14b5104924e09f3e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-45d5600d06daa65c636d173fc2658cddb67f606f3fa58c752c96f36022342c82 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cbef5aebcc22974e77cace1461d7e8c586c10808ff47410e04a8bd6d096db979 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5f1796327a533f04533b784cf8e1700a90e7c3505c4c7b0b61069a960064de61 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f6e425fb | sha256-72818adeaa505c9445d13bb2cf956ab1e3aca27ea41685872844010ae7afdc48 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f6e425fb | sha256-6caef9a12105bfc0ec7723d3c67e6721a8f34f6d14669c0f61175e7a2a8a3f12 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f6e425fb | sha256-72818adeaa505c9445d13bb2cf956ab1e3aca27ea41685872844010ae7afdc48 |

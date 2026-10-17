"""Fine-cell absence certificate layered after the frozen observation reducer.

joint.state stays byte-identical (it is part of the learned-model signature); the
runner, public executor and both replayers call certify_fine right after
update_observation so that live runs and audits agree on the exit certificate.
"""

from dataclasses import replace

from rf_searchlab.algorithms.route.coverage import fine_certified_absent
from rf_searchlab.algorithms.shared.state import update_observation
from rf_searchlab.core.domain import Certificate, Envelope, Response, State, Status

KIND = "full_fine_cell_negative_cover"


def certify_fine(state: State) -> State:
    beliefs = list(state.beliefs)
    changed = False
    for i, b in enumerate(beliefs):
        if b.status == Status.UNKNOWN and fine_certified_absent(state, b.channel):
            evidence = tuple(o.request_id for o in b.observations if o.result == "no_signal")
            beliefs[i] = replace(
                b,
                status=Status.ABSENT,
                certificate=Certificate(KIND, b.channel, evidence, b.version),
            )
            changed = True
    return replace(state, beliefs=tuple(beliefs)) if changed else state


def update_observation_fine(state: State, envelope: Envelope, response: Response) -> State:
    return certify_fine(update_observation(state, envelope, response))

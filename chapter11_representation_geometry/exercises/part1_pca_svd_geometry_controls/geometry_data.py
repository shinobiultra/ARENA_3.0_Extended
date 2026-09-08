"""Readable, frozen calendar prompts; no model calls or fitting live in this file."""
from __future__ import annotations

import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
NAMES={
    'weekday':('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
    'month':('January','February','March','April','May','June','July','August','September','October','November','December'),
}
NUMBERS=('Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Eleven','Twelve')


def load_protocol():
    return json.loads((HERE/'geometry_protocol.json').read_text())


def calendar_rows(family,split,protocol=None):
    """One row per start, offset and template, including the exact semantic span."""
    protocol=protocol or load_protocol()
    names=NAMES[family]
    if split=='final':
        templates=protocol['final_templates'][family]
        offsets=range(1,len(names)+1)
    elif split in ('train','development'):
        templates=["Let's do some days of the week math. {interval} from {name} is" if family=='weekday'
                   else "Let's do some calendar math. {interval} from {name} is"]
        offsets=protocol['train_offsets' if split=='train' else 'development_offsets'][family]
    else:
        raise ValueError('Choose train, development or final.')
    result=[]
    for template_index,template in enumerate(templates):
        for start,name in enumerate(names):
            for offset in offsets:
                unit='day' if family=='weekday' else 'month'
                # Author's month task uses "One months"; retain it for training/development parity.
                plural=offset!=1 or (family=='month' and split!='final')
                interval=NUMBERS[offset]+' '+unit+('s' if plural else '')
                prompt=template.format(interval=interval,name=name)
                left=prompt.rindex(name)
                result.append({'id':f'{split}/{family}/template{template_index}/start{start}/offset{offset}',
                               'family':family,'split':split,'template_index':template_index,'start':start,'offset':offset,
                               'answer':(start+offset)%len(names),'prompt':prompt,'source_span':[left,left+len(name)]})
    return result


def paired_donor_indices(rows,shift,period):
    """Match a counterfactual at the same template and offset, changing only start."""
    lookup={(row['template_index'],row['start'],row['offset']):i for i,row in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('Calendar rows must be unique.')
    return [lookup[row['template_index'],(row['start']+shift)%period,row['offset']] for row in rows]

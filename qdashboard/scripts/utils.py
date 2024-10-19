#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dash import  dcc


# =============================================================================
# radio to switch between linear and log scale 
# =============================================================================


def LinearLogRadio(id_str):
    return dcc.RadioItems( ['Linear', 'Log'], 'Linear', inline=True, 
                   id=id_str,
                    inputStyle={"margin-left": "5px", 
                                "margin-right": "2px",},
                    labelStyle={"font-size":"9pt"},
                    className='radio'
                    )

    
    
y_ordered = ['Alph_Rhod_Rhod_azot', 'Alph_Rhod_Rhod_mari', 
             'Alph_Holo_Holo_acum', 'Alph_Holo_Holo_cary', 
             'Alph_Acet_Acet_past', 'Alph_Acet_Acet_lova', 
             
             'Gamm_Card_Wohl_larv', 
             'Gamm_Ente_Esch_coli', 'Gamm_Ente_Yers_pseu',
             'Gamm_Ente_Idio_seos',  'Gamm_Ente_Idio_pisc',
             'Gamm_Ente_Pseu_arab', 'Gamm_Ente_Pseu_phen'
             ]


def order_by_species(data):
    data['yo'] = data.index.to_list()
    data.yo = data.yo.astype("category")
    data.yo = data.yo.cat.set_categories(y_ordered)
    data = data.sort_values(['yo'])
    data = data.drop(columns=['yo'])
    return data


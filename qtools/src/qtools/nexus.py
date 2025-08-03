'''
convert a distance matrix to nexus file and plot it with SplitsTree4
(https://software-ab.informatik.uni-tuebingen.de/download/splitstree4/manual.pdf)
    
USAGE:
    imgformat:  One of ['jpg', 'eps', 'svg', 'png', 'pdf'].
    plot_now:   [True, False]. If True, plot directly to same dir as outfile
    cmdfile:    Filename to which splitstree-commands should be written. 
    cmdmode:    One of ['w', 'a']. write to a new cmdfile (w) or append existing file (a)

Usage of the cmdfile with splitstree:
    ~/splitstree4/SplitsTree -g -c cmdfile
'''

import os

def matrix2nexus(matrix,
                 taxa,
                 nexusfile,
                 imgformat='png',
                 plot_now=True,
                 cmdfile=False,
                 cmdmode='a',
                 splitstree_location='~/splitstree4/SplitsTree'):
    
    # convert taxa names to nexus header
    num_of_taxa = len(matrix)
    taxa_list = ['[' + str(x + 1) + ']' for x in range(num_of_taxa)]
    taxa_list = ' '.join([x for y in zip(taxa_list, taxa) for x in y])

    # convert matrix to nexus format
    b = [matrix[x, :x + 1] for x in range(num_of_taxa)]
    b = [str(x).replace('\n', '').strip('[]') for x in b]
    matrix_out = '\n'.join([' '.join([x, y]) for x, y in zip(taxa, b)])

    # arrange file
    lines = {
        1: '#NEXUS',
        2:
        f'BEGIN taxa; DIMENSIONS ntax={num_of_taxa}; TAXLABELS {taxa_list} ; END;',
        3:
        f'BEGIN distances; DIMENSIONS ntax={num_of_taxa}; FORMAT triangle=LOWER diagonal labels missing=? ;',
        4: 'MATRIX ',
        5: matrix_out,
        6: '; END;',
    }
    text_out = '\n'.join(lines.values())

    # write out the nexus file
    with open(nexusfile, 'w') as outf:
        outf.write(text_out)

    # execute splitstree directly
    imagefile = nexusfile.replace('.nex', '.' + imgformat.lower())
    if plot_now:
        outline = f"'EXECUTE file={nexusfile}; EXPORTGRAPHICS format={imgformat.upper()} file={imagefile} REPLACE=yes; QUIT;'"
        os.system(f"{splitstree_location} -g -v -x " + outline +
                  ' > /dev/null 2>&1')
    # make splitstree command file
    if cmdfile:
        with open(cmdfile, cmdmode) as outf:
            outline = f'EXECUTE file={nexusfile}\nEXPORTGRAPHICS format={imgformat.upper()} file={imagefile} REPLACE=yes\n'
            outf.write(outline + '\n')

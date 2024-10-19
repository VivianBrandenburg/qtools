import os
from datetime import datetime
import itertools

def update_dir(outdir_1):
    """
    Takes a path as string. Creates a new subdirectory from datetime.now and adds it to the provided path.  
    Returns 'given_path/new_subdirectory/'
    """
    outdir_1 = outdir_1.strip('/') + '/'
    dt = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    newdir = outdir_1 + dt + '/'
    create_dir(newdir)
    create_dir(newdir+'/nexus/')
    create_dir(newdir+'/weights/')
    return newdir



def create_dir(newdir):
    """
    Checks whether a directory already exists and creates it if it does not exist. 
    """
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    return newdir


def iter_over_vars(**kwargs):
    """
    Takes a list of arguments as variable=[values]. 
    Returns it as iterable dictionary [{variable:value1}, {variable:value2}].
    """
    keys, values = zip(*kwargs.items())
    variables = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # renamed 'runtype' to 'mode', this is for backward compatibility
    # and might also be needed for compatibility with the dashboard
    for x in variables: 
        if 'mode' in x.keys(): x.update({'runtype':x['mode']}) 
    
    return variables

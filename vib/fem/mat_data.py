from collections import defaultdict


# Materials for different physical types(idx)
material = defaultdict(
    lambda : { # default value
        'nu' : 0.3,
        'E' : 1e3
    },
    {
        100 : { # interior
            'nu' : 0.3,
            'E' : 1.0
        },
        200 : { # exterior
            'nu' : 0.3,
            'E' : 5.0
        }
    }
)

load = {
    500 : {
        'val' : -2, # -2 for template.msh
        'dir' : 1
    }
}

boundary = {
    300 : { # x-dir
        'val' : -1,
        'dir' : 0
    },
    400 : { # y-dir
        'val' : -2,
        'dir' : 1
    }
}

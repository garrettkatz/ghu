"""
Echo input (rinp) at output (rout)
"""
from ghu import GatedHebbianUnit, DefaultController


if __name__ == "__main__":
    
    layer_sizes = {"rinp": 3, "rout":3}
    pathways = [(0,("rinp","rinp")), (1, ("r1","rinp"))]
    hidden_size = 5
    dc = DefaultController(layer_sizes, pathways, hidden_size)
    print(dc)

    ghu = GatedHebbianUnit(layer_sizes, pathways, dc)
    print(ghu)



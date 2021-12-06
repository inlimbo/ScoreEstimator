class Trees:
    def __init__(self):
        self._tree_list = []
    def add_tree(self, tree):
        self._tree_list.append(tree)
    def print_trees(self):
        for x in self._tree_list:
            print(x)
    


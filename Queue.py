class Queue:
    def __init__(self, size):
        self.size = size
        self.array = []

    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, index):
        return self.array[index]

    def add(self, el):
        if len(self.array) + 1 > self.size:
            self.array.pop(0)
        self.array.append(el)

    def clear(self):
        self.array = []

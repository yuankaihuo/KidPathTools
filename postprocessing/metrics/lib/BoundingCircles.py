from BoundingCircle import *
from utils import *


class BoundingCircles:
    def __init__(self):
        self._boundingCircles = []

    def addBoundingCircle(self, bb):
        self._boundingCircles.append(bb)

    def removeBoundingCircle(self, _boundingCircle):
        for d in self._boundingCircles:
            if BoundingCircle.compare(d, _boundingCircle):
                del self._boundingCircles[d]
                return

    def removeAllBoundingCircles(self):
        self._boundingCircles = []

    def getBoundingCircles(self):
        return self._boundingCircles

    def getBoundingCircleByClass(self, classId):
        BoundingCircles = []
        for d in self._boundingCircles:
            if d.getClassId() == classId:  # get only specified bounding box type
                BoundingCircles.append(d)
        return BoundingCircles

    def getClasses(self):
        classes = []
        for d in self._boundingCircles:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingCirclesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingCircles if d.getBBType() == bbType]

    def getBoundingCirclesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingCircles if d.getImageName() == imageName]

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingCircles)
        count = 0
        for d in self._boundingCircles:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def clone(self):
        newBoundingCircles = BoundingCircles()
        for d in self._boundingCircles:
            det = BoundingCircle.clone(d)
            newBoundingCircles.addBoundingCircle(det)
        return newBoundingCircles

    def drawAllBoundingCircles(self, image, imageName):
        bbxes = self.getBoundingCirclesByImageName(imageName)
        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
        return image

    # def drawAllBoundingCircles(self, image):
    #     for gt in self.getBoundingCirclesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingCirclesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image

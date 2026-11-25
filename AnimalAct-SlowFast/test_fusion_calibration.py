import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(r'C:\Users\ADMIN\Desktop\作业与项目\特选论题1\项目\AnimalAct-SlowFast')))
from fusion_calibration import fit,cross_validate
from metrics import evaluate


class FusionTests(unittest.TestCase):
    def test_class_experts_can_be_recovered_without_oracle_test_labels(self):
        target=np.array([[True,True],[False,False]]*30)
        base=np.array([[.9,.2],[.7,.8]]*30,dtype=np.float32)
        new=np.array([[.1,.9],[.9,.3]]*30,dtype=np.float32)
        classes=[dict(name='a',global_id=1),dict(name='b',global_id=2)]
        weights,threshold=fit(target,base,new,classes,0)
        self.assertEqual(weights[0],0)
        self.assertGreater(weights[1],.5)
        scores=(1-weights)*base+weights*new
        self.assertEqual(evaluate(target,scores,classes,threshold)['mAP'],1.)

    def test_identical_models_remain_finite_convex_and_stable(self):
        target=np.array([[True,True],[False,False]]*15)
        scores=np.array([[.9,.8],[.1,.2]]*15,dtype=np.float32)
        classes=[dict(name='a',global_id=1),dict(name='b',global_id=2)]
        selected,table=cross_validate(target,scores,scores,classes,[str(i//2) for i in range(30)])
        self.assertIsNone(selected['regularizer'])
        weights,_=fit(target,scores,scores,classes,10)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(((weights>=0)&(weights<=1)).all())


if __name__=='__main__':unittest.main()

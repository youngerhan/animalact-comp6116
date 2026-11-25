# Packaging adaptations

- Original sources and trained weights are preserved outside this bundle.
- Run settings in this bundle store paths relative to AnimalAct-SlowFast. The bundled clips.read_json resolves these paths in memory; original settings are in original-settings/.
- The bundled MMAction2 requirements/build.txt names opencv-python instead of opencv-contrib-python, matching the OpenCV distribution actually used by this project. No network architecture or pretrained weights were modified. Original requirement file is preserved here.
- pypdf was repacked from the installed pure-Python distribution because its original wheel was unavailable. This package is used for documents, not model computation.
- runtime/ contains Python 3.11.18 and its standard library/DLLs. Third-party packages are installed offline from wheelhouse/ after extraction. Original optimizer checkpoints last.pth are omitted.
- Legacy source is archived for auditing earlier random-crop behavior; current-code retraining is not guaranteed to reproduce the earlier numerical results exactly.

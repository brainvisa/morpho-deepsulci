
_doc_path = None

def _init_doc_path():
    global _doc_path
    import os
    import deepsulci

    # use axon config for version
    try:
        import brainvisa.config as bv_config
        axon_version = bv_config.shortVersion
    except:
        axon_version = '4.6'

    opd = os.path.dirname
    p = os.path.join(
        opd(opd(opd(deepsulci.__file__))),
        'share/doc/morpho-deepsulci-%s/dev_doc/process_docs/deepsulci/' \
          'pattern_classification' % axon_version)
    if os.path.exists(p):
        _doc_path = p
        return _doc_path
    _doc_path = 'http://brainvisa.info/morpho-deepsulci-%s/dev_doc/' \
        'process_docs/deepsulci/pattern_classification' % axon_version
    return _doc_path

_init_doc_path()


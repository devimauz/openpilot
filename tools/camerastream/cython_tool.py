from SCons.Script import Builder, Action

def generate(env):
    # Create a new Builder for Cython
    cython_action = Action('cython ${CYTHONFLAGS} -o $TARGET $SOURCE', 'Cythonizing $SOURCE')
    cython_builder = Builder(action=cython_action, suffix='.c', src_suffix='.pyx')
    
    # Add the Builder to the environment
    env.Append(BUILDERS={'Cython': cython_builder})
    env['CYTHON'] = 'cython'
    env['CYTHONFLAGS'] = ''

def exists(env):
    return env.Detect('cython')

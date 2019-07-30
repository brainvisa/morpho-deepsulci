cmake_minimum_required( VERSION 2.6 )
find_package( brainvisa-cmake REQUIRED )
BRAINVISA_PROJECT()
enable_testing()

BRAINVISA_FIND_PACKAGE( axon REQUIRED )
BRAINVISA_FIND_PACKAGE( morphologist-gpl QUIET )
BRAINVISA_DEPENDENCY( RUN DEPENDS "axon" RUN
                      ">= ${axon_VERSION_MAJOR}.${axon_VERSION_MINOR}" )
BRAINVISA_DEPENDENCY( RUN RECOMMENDS "morphologist-gpl" RUN
                      ">= ${morphologist-gpl_VERSION_MAJOR}.${morphologist-gpl_VERSION_MINOR}" )
BRAINVISA_DEPENDENCY( DEV DEPENDS "${PROJECT_NAME}" RUN "= ${${PROJECT_NAME}_VERSION}" )

BRAINVISA_COPY_PYTHON_DIRECTORY( "${CMAKE_CURRENT_SOURCE_DIR}/brainvisa"
                                 ${PROJECT_NAME} )
BRAINVISA_COPY_PYTHON_DIRECTORY( "${CMAKE_CURRENT_SOURCE_DIR}/python"
                                 ${PROJECT_NAME} )

BRAINVISA_COPY_DIRECTORY( "${CMAKE_CURRENT_SOURCE_DIR}/share/foms"
                          share/foms
                          ${PROJECT_NAME} )

# BRAINVISA_COPY_DIRECTORY( "${CMAKE_CURRENT_SOURCE_DIR}/doc"
#                           "share/doc/morphologist-${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
#                           ${PROJECT_NAME}-usrdoc )

# brainvisa_add_test(morphologist-tests "${PYTHON_EXECUTABLE_NAME}" -m brainvisa.tests.test_morphologist --no-ann TESTREF)
# brainvisa_add_test(morphologist-test-capsul "${PYTHON_EXECUTABLE_NAME}" -m morphologist.tests.test_morphologist_capsul)
# set_tests_properties(morphologist-test-capsul PROPERTIES DEPENDS morphologist-tests)

find_package( Sphinx )
BRAINVISA_GENERATE_SPHINX_DOC( "sphinx"
  "share/doc/morpho-deepsulci-${BRAINVISA_PACKAGE_VERSION_MAJOR}.${BRAINVISA_PACKAGE_VERSION_MINOR}/dev_doc" )

# BRAINVISA_CREATE_CMAKE_CONFIG_FILES()


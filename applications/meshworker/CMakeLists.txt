FILE(GLOB sources *.cc)
GET_FILENAME_COMPONENT(prefix ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF(PVS-Analysis)
  INCLUDE(${CMAKE_SOURCE_DIR}/tools/PVS-Studio.cmake)
  SET(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
ENDIF()
FOREACH(ccfile ${sources})
  GET_FILENAME_COMPONENT(file ${ccfile} NAME_WE)
  SET(target ${file})
  ADD_EXECUTABLE(${target} ${ccfile})
  SET_TARGET_PROPERTIES(${target} PROPERTIES OUTPUT_NAME ${file})
  DEAL_II_SETUP_TARGET(${target})

  IF(PVS-Analysis)
    pvs_studio_add_target(TARGET analyze_${target} ALL
                          ANALYZE ${target}
                          OUTPUT FORMAT errorfile
                          CXX_FLAGS ${DEAL_II_CXX_FLAGS}
                          LOG ${target}.plog
                          CONFIG "${CMAKE_SOURCE_DIR}/PVS-Studio.cfg")
  ENDIF()
ENDFOREACH()

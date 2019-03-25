import java.io.PrintWriter

object rui_chen_task1 {
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()
    val inputPath = args(0)
    val output = new PrintWriter(args(1))

    val similarities = new getSimilarities().getFrom(inputPath, false)
    output.write("business_id_1, business_id_2, similarity\n")
    for (pair <- similarities) {
      output.write(pair._1 + "," + pair._2 + "\n")
    }
    output.close()
    val endTime = System.currentTimeMillis()
    println("Duration: " + (endTime - startTime) / 1000.0)
  }
}

#include "lob.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_civitas_lob, m) {
  m.doc() = "Civitas C++ Limit Order Book using pybind11";

  py::class_<RuleConfig>(m, "RuleConfig")
      .def(py::init<>())
      .def_readwrite("commission_rate", &RuleConfig::commission_rate)
      .def_readwrite("stamp_duty_rate", &RuleConfig::stamp_duty_rate)
      .def_readwrite("min_price_tick", &RuleConfig::min_price_tick)
      .def_readwrite("min_trade_unit", &RuleConfig::min_trade_unit)
      .def_readwrite("board_lot", &RuleConfig::board_lot)
      .def_readwrite("enforce_min_trade_unit", &RuleConfig::enforce_min_trade_unit)
      .def_readwrite("enforce_board_lot", &RuleConfig::enforce_board_lot)
      .def_readwrite("allow_odd_lots", &RuleConfig::allow_odd_lots)
      .def_readwrite("strict_queue_timestamps", &RuleConfig::strict_queue_timestamps)
      .def_readwrite("timestamp_precision", &RuleConfig::timestamp_precision);

  py::class_<Order>(m, "Order")
      .def(py::init<>())
      .def_readwrite("order_id", &Order::order_id)
      .def_readwrite("agent_id", &Order::agent_id)
      .def_readwrite("timestamp", &Order::timestamp)
      .def_readwrite("symbol", &Order::symbol)
      .def_readwrite("side", &Order::side)
      .def_readwrite("order_type", &Order::order_type)
      .def_readwrite("price", &Order::price)
      .def_readwrite("quantity", &Order::quantity)
      .def_readwrite("status", &Order::status)
      .def_readwrite("filled_qty", &Order::filled_qty)
      .def_property_readonly("remaining_qty", &Order::remaining_qty)
      .def_property_readonly("is_filled", &Order::is_filled);

  py::class_<Trade>(m, "Trade")
      .def(py::init<>())
      .def_readwrite("trade_id", &Trade::trade_id)
      .def_readwrite("price", &Trade::price)
      .def_readwrite("quantity", &Trade::quantity)
      .def_readwrite("maker_id", &Trade::maker_id)
      .def_readwrite("taker_id", &Trade::taker_id)
      .def_readwrite("maker_agent_id", &Trade::maker_agent_id)
      .def_readwrite("taker_agent_id", &Trade::taker_agent_id)
      .def_readwrite("timestamp", &Trade::timestamp)
      .def_readwrite("buyer_fee", &Trade::buyer_fee)
      .def_readwrite("seller_fee", &Trade::seller_fee)
      .def_readwrite("seller_tax", &Trade::seller_tax);

  py::class_<LimitOrderBook>(m, "LimitOrderBook")
      .def(py::init<std::string, RuleConfig>(), py::arg("symbol"), py::arg("rule_config") = RuleConfig())
      .def("add_order", &LimitOrderBook::add_order)
      .def("cancel_order", &LimitOrderBook::cancel_order)
      .def("get_best_bid", &LimitOrderBook::get_best_bid)
      .def("get_best_ask", &LimitOrderBook::get_best_ask)
      .def("get_depth", &LimitOrderBook::get_depth)
      .def("clear", &LimitOrderBook::clear)
      .def("set_rule_config", &LimitOrderBook::set_rule_config)
      .def("get_rule_config", &LimitOrderBook::get_rule_config)
      .def("add_order_exploded",
           [](LimitOrderBook &self, std::string order_id, std::string agent_id,
              double timestamp, std::string symbol, std::string side,
              std::string order_type, double price, double quantity) {
             Order order;
             order.order_id = order_id;
             order.agent_id = agent_id;
             order.timestamp = timestamp;
             order.symbol = symbol;
             order.side = side;
             order.order_type = order_type;
             order.price = price;
             order.quantity = quantity;
             order.filled_qty = 0;
             order.status = "pending";

             std::vector<Trade> trades = self.add_order(order);

             py::list py_trades;
             for (auto &trade : trades) {
               py_trades.append(py::make_tuple(
                   trade.trade_id, trade.price, trade.quantity, trade.maker_id,
                   trade.taker_id, trade.maker_agent_id, trade.taker_agent_id,
                   trade.timestamp, trade.buyer_fee, trade.seller_fee,
                   trade.seller_tax));
             }

             return py::make_tuple(order.filled_qty, order.status, py_trades);
           });
}
